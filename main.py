import gzip
import imghdr
import io

import cv2
import numpy as np
from PIL import Image
import functools
import json
import lzma
import math
import multiprocessing as mp
import os
import re
import urllib
import warnings

import wikitextparser
import pypandoc
import pandas as pd

from dateutil import parser as date_parser
from dateutil.parser import ParserError
from lxml import etree
from sqlalchemy import create_engine, MetaData, Table, Column, ForeignKey, String, Boolean

# Monkeypatch bug in imagehdr
from imghdr import tests

ENCODING = 'utf8'
EXTERNAL_LINK_MAPPINGS = {
    'booth': 'booth.pm',
    'discord': 'discord.gg',
    'instagram': 'instagram.com/',
    'facebook': 'facebook.com/',
    'marshmallow': 'marshmallow-qa.com',
    'patreon': 'patreon.com',
    'pixiv': 'pixiv.net/',
    'twitch': 'twitch.tv',
    'twitter': 'twitter.com/',
}
WIKI_PAGE_BASE_LINK = 'https://virtualyoutuber.fandom.com/wiki/'
WIKI_LINK = WIKI_PAGE_BASE_LINK + 'Virtual_YouTuber_Wiki'

PILLOW_THUMBNAIL_MAX_SIZE = 400, 400
CV2_MAX_HEIGHT = 400


def test_jpeg1(h, f):
    """JPEG data in JFIF format"""
    if b'JFIF' in h[:23]:
        return 'jpeg'


JPEG_MARK = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06' \
            b'\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f'


def test_jpeg2(h, f):
    """JPEG with small header"""
    if len(h) >= 32 and 67 == h[5] and h[:32] == JPEG_MARK:
        return 'jpeg'


def test_jpeg3(h, f):
    """JPEG data in JFIF or Exif format"""
    if h[6:10] in (b'JFIF', b'Exif') or h[:2] == b'\xff\xd8':
        return 'jpeg'


tests.append(test_jpeg1)
tests.append(test_jpeg2)
tests.append(test_jpeg3)


def get_external_links(external_links, is_agency):
    _ext_links = {}
    for ext_link in external_links:
        text = ext_link.text
        url = ext_link.url

        # Official Website
        if text is not None and (
                text.strip().lower() == 'official site' or text.strip().lower() == 'official website'):
            _ext_links['official_site'] = url
        # Youtube Channel
        channels = re.findall('(youtube.com|youtu.be)/(channel|c)/', url)
        if len(channels) > 0:
            if _ext_links.get('official_channel') is not None:
                warnings.warn("More than one youtube channel detected: " + ext_link.string)
                if text.strip().lower().find('sec') > 0 or text.strip().lower().find('sub') > 0:
                    _ext_links['secondary_official_channel'] = url
            else:
                _ext_links['official_channel'] = url
        # Twitter/Facebook/Instagram/Pixiv, etc
        for k, v in EXTERNAL_LINK_MAPPINGS.items():
            channels = re.findall(v, url)
            if k == 'twitter' and url.find('/status') > -1:  # skip tweets
                continue
            if len(channels) > 0:
                if _ext_links.get(k) is not None:
                    warnings.warn("More than one " + k + " link detected: " + ext_link.string)
                    # Some agencies have their talents' twitter listed
                    if is_agency:
                        _ext_links[k] = None
                else:
                    _ext_links[k] = url

    return _ext_links


def get_categories(wikitext):
    categories = []
    for wikilink in wikitext.wikilinks:
        splits = wikilink.target.strip().split(':')
        if len(splits) > 1 and splits[0].strip().lower() == 'category':
            categories.append(splits[1].strip().lower())
    return categories


def parse_image_urls(file):
    """
    Parse image urls from dump and save to a dictionary
    :param file: image dump file
    :return:
    """

    def is_valid_url(url):
        # URL validator from old 1.3.x branch of django
        # Does not support punycode, ipv6, etc. But enough here
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url) is not None

    _img_urls = {}
    with open(file, 'r', encoding='utf8') as f:
        images = f.readlines()
        for image in images:
            segments = image.strip().split('\t')
            filename = segments[0]
            for seg in segments:
                # Assuming only one url
                if is_valid_url(seg):
                    idx = seg.rfind('/revision')
                    if idx > 0:
                        seg = seg[:idx]
                    _img_urls[filename.strip().lower()] = seg
                    _img_urls[filename.strip().lower().replace(' ', '_')] = seg

    return _img_urls


def get_title(page, xmlns):
    titles = page.xpath('.//x:title', namespaces={'x': xmlns})
    assert len(titles) == 1
    title = titles[0].text
    return title


def parse_single_page(xml_path, title_to_parse, xmlns, image_urls, output_dict):
    """
    Subroutine for parsing xml dump file.
    Simply sopy parameters from outside loop, assuming we have enough memory.
    :param page: A single page
    :param xmlns: Default namespace in xml file
    :param image_urls: A dictionary containing image url dumps
    """

    # We cannot pass ElementTree directly between processes because of pickle
    # XML parser
    parser = etree.XMLParser()
    tree = etree.parse(xml_path, parser=parser)
    pages = tree.xpath('//x:page', namespaces={'x': xmlns})

    for page in pages:
        title = get_title(page, xmlns)
        if title == title_to_parse:
            # Easier to insert title name here for later insertion to SQL
            _info_dict = {
                'name': title,
                'is_agency': False,
                'profile_img': '',
                'profile_img_link': '',
                'intro': '',
                'character': {
                    'name': title,
                },
                'ext_links': {
                    'name': title,
                }
            }

            texts = page.xpath('.//x:revision/x:text', namespaces={'x': xmlns})
            assert len(texts) == 1  # There shall only be one text section
            # print("Title: " + title)

            for text in texts:
                inner_text = text.text
                if inner_text.startswith('#REDIRECT'):
                    return
                wikitext = wikitextparser.parse(inner_text)
                sections = wikitext.get_sections(include_subsections=False)
                categories = get_categories(wikitext)
                if 'forum' in categories:
                    return
                if 'agency' in categories:
                    is_agency = True
                else:
                    is_agency = False
                _info_dict['is_agency'] = is_agency

                # Parse first section
                intro_section = sections[0]
                assert intro_section.title is None
                wikilinks = intro_section.wikilinks

                intro_text = remove_links(intro_section.string, intro_section)
                _info_dict['intro'] = markup_to_markdown(intro_text)

                # Parse template for basic information
                for template in intro_section.templates:
                    parse_character_section(_info_dict, image_urls, template)

                parse_title_generate_credit(_info_dict, intro_section)

                # Profile images
                if _info_dict.get('profile_img', '') == '':
                    for wikilink in wikilinks:
                        get_profile_image(_info_dict, image_urls, wikilink.target)

                # Parse external links
                for section in sections:
                    if section.title is not None and \
                            (section.title.strip().lower() == 'external links' or section.title.strip(
                            ).lower() == 'media'):
                        _info_dict['ext_links'].update(get_external_links(section.external_links, is_agency))

                output_dict[title] = _info_dict
                # pprint(_info_dict)


def skip_arg_name(name):
    skip_names = [
        '1', 'channel', 'discord', 'image2', 'image_color',
        'social_media', 'twitch',
    ]
    if name in skip_names:
        return True
    return False


def map_arg_name(name):
    mapping = {
        'age.': 'age',
        'character design': 'character_designer',
        'chinese_horoscope': 'zodiac_sign',
        'emoij': 'emoji',
        'heitght': 'height',
        'nick_name': 'nickname(s)',
        'nickname': 'nickname(s)',
        'nicknames': 'nickname(s)',
        'official_websit e': 'official_website',
        'official_websit_e': 'official_website',
        'stream_start_date': 'debut_date',
    }

    # Remove numbers at end
    name = re.sub(r'\d+$', '', name)

    # convert spaces to underscores
    name = name.replace(' ', '_')

    # Map the rest
    if name in list(mapping.keys()):
        name = mapping[name]

    return name


def parse_character_section(_info_dict, image_urls, template):
    """
    The known field names in character sections are:
    ['1', 'affiliation', 'age', 'age.', 'artist', 'birthday',
     'blood type', 'blood_type', 'caption', 'caption1', 'channel',
      'character design', 'character designer', 'character modeler',
       'character_designer', 'character_modeler', 'chinese_horoscope',
        'debut_date', 'discord', 'emoij', 'emoji', 'family', 'fans',
         'full_name', 'gender', 'height', 'heitght', 'image2', 'image_color',
          'language', 'mbti type', 'name', 'name_stand', 'names', 'nick_name',
           'nickname', 'nickname(s)', 'nicknames', 'occupation',
            'official_websit e', 'official_websit_e', 'official_website',
             'original_name', 'power', 'race', 'social_media', 'species',
              'stream_start_date', 'team_affiliation', 'title1', 'twitch',
               'weight', 'zodiac sign', 'zodiac_sign']
    Drop some (like '1') and normalize others (like 'age' and 'age.').
    :param _info_dict: The dict to output to
    :param image_urls: Images data from dump
    :param intro_section: Parsed introduction section
    :return:
    """
    if template.name.strip().lower() == 'character':
        for argument in template.arguments:
            arg_name = argument.name.strip().lower()
            arg_name_mapped = map_arg_name(arg_name)
            if skip_arg_name(arg_name):
                continue
            # Process avatar images
            if arg_name == 'image1':
                get_profile_image(_info_dict, image_urls, argument.value)
            else:
                value = remove_links(argument.value, argument)
                _info_dict['character'][arg_name_mapped] = markup_to_markdown(value)


def parse_title_generate_credit(_info_dict, intro_section):
    display_title = _info_dict['name']

    for parser_function in intro_section.parser_functions:
        if parser_function.name.strip().lower() == 'displaytitle':
            assert len(parser_function.arguments) == 1
            display_title = parser_function.arguments[0].value

    _info_dict['display_title'] = display_title
    _info_dict['page_link'] = WIKI_PAGE_BASE_LINK + urllib.parse.quote(
        _info_dict['name'].replace(' ', '_'))
    # Simpler to use Markdown
    _info_dict['credit'] = 'This page uses material from the ["{}"]({}) article on the [Virtual YouTuber Wiki]({}) at ' \
                           'Fandom and is licensed under the [Creative Commons Attribution-Share Alike License](' \
                           'https://creativecommons.org/licenses/by-sa/3.0/). '.format(
        _info_dict['name'], _info_dict['page_link'], WIKI_LINK)


def get_profile_image(info_dict, image_urls, string):
    groups = re.match(r'^(?:\[\[)?(?:File:)?(.*(?:\.gif|\.webp|\.jpg|\.jpeg|\.png))(?:]])?$', string.strip())
    if groups is not None and len(groups.groups()) > 0:
        info_dict['profile_img'] = groups.groups()[0].strip()
        info_dict['profile_img_link'] = image_urls[info_dict['profile_img'].strip().lower()]


def should_skip_page(title):
    exclude_string = [
        '/Gallery',
        '/Discography',
        '/D&D',
        '/Visitor',
        '/Relation',
        'Talk:',
        'Draft:',
        'Template:',
        'User:',
        'File:',
        'Help:',
        'MediaWiki:',
        'Category:',
        'Message Wall:',
        'Thread:',
        'blog:',
        'talk:',
        'comment:',
        'Wiki:',
        'Wall:',
        'Greeting:',
        'Module:',
        'Sandbox:',
    ]
    for s in exclude_string:
        if s.lower() in title.replace(' ', '').lower():
            return True
    return False


def parse_xml(dump_root, prefix, multiprocess=True):
    """
    Parse mediawiki dump xml and return a dictionary containing necessary information.
    Assuming no history.
    :return: A JSON formatted string containing parsed information
    """
    # XML parser
    xml_path = os.path.join(dump_root, prefix + '-current.xml')
    parser = etree.XMLParser()
    tree = etree.parse(xml_path, parser=parser)
    image_urls = parse_image_urls(os.path.join(dump_root, prefix + '-images.txt'))

    # Debug
    with open('out/debug.out', 'w', encoding='utf8') as f:
        f.write(etree.tostring(tree.getroot(), encoding='unicode'))

    # find namespace
    xmlns = tree.getroot().nsmap.copy().pop(None)
    pages = tree.xpath('//x:page', namespaces={'x': xmlns})

    # Run parsing on pages
    mp_dict = {}

    # Starting multiprocess pool
    if multiprocess:
        pool = mp.Pool(processes=mp.cpu_count())
        manager = mp.Manager()
        mp_dict = manager.dict()

    for page in pages:
        title = get_title(page, xmlns)
        if should_skip_page(title):
            continue

        if multiprocess:
            pool.apply_async(parse_single_page, (
                xml_path, title, xmlns, image_urls, mp_dict))
        else:
            try:
                parse_single_page(xml_path, get_title(page, xmlns), xmlns, image_urls, mp_dict)
            except Exception as e:
                print(e)
                print(e.__traceback__)
    if multiprocess:
        pool.close()
        pool.join()

    # Return json formatted string
    json_out = json.dumps(mp_dict.copy(), ensure_ascii=False, allow_nan=False, indent=2)
    print(json_out)
    return json_out


def remove_links(text, section, type='both'):
    if type == 'wikilink' or type == 'both':
        for wikilink in section.wikilinks:
            # Do not replace wikilink without namespace
            replace_str = ''
            groups = re.match(r'^\[\[([^:]*)(:[^:]*)*]]$', wikilink.string)
            if groups is not None:
                groups = [x for x in groups.groups() if x is not None]
                if len(groups) == 1:
                    new_str = groups[0]
                    groups = re.match(r'([^|]*)(\|[^|]*)*', new_str)
                    if groups is not None:
                        groups = [x for x in groups.groups() if x is not None]
                        if len(groups) == 1:
                            replace_str = groups[0]
                        elif len(re.findall(r'\|', new_str)) == 1:
                            replace_str = groups[0]

            text = text.replace(wikilink.string, replace_str)

    # if type == 'external_link' or type == 'both':
    #     for ext_link in section.external_links:
    #         link_text = ext_link.url if ext_link.text is None or ext_link.text == '' else ext_link.text
    #         markdown = r'[{}]({})'.format(link_text, ext_link.url)
    #         text = text.replace(ext_link.string, markdown)
    return text.strip()


def markup_to_markdown(text):
    return pypandoc.convert_text(
        text, 'markdown_github+smart', format='mediawiki+smart', extra_args=['--wrap=none']).strip()


def image_resize(image, height=None, width=None, interp=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=interp)

    # return the resized image
    return resized


def json_to_sql(data, database_path, img_base_path,
                embed_image=False, include_app_tables=True,
                use_lzma=False):
    def map_image_file(filename, base_path):
        if use_lzma:
            compressor = lzma.LZMACompressor(check=lzma.CHECK_CRC32, preset=lzma.PRESET_EXTREME)
        else:
            bytes_out = io.BytesIO()
            gzipFile = gzip.GzipFile(fileobj=bytes_out, mode="wb")
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            # convert png, etc. to jpg
            with open(full_path, 'rb') as f:
                bin_data = f.read()
                img_format = imghdr.what(full_path)
                if img_format is None:
                    img_format = 'None'
                # with Image.open(full_path) as img:
                #     if img.format.lower() != 'jpeg' or img.format.lower() != 'jpg':
                if img_format != 'jpeg' and img_format != 'jpg':
                    img_stream = io.BytesIO(bin_data)
                    img_stream.seek(0)
                    if img_format != 'gif':
                        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                        img = image_resize(img, height=CV2_MAX_HEIGHT, interp=cv2.INTER_LANCZOS4)
                        is_success, buffer = cv2.imencode(
                            ".jpg", img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85,
                             int(cv2.IMWRITE_JPEG_PROGRESSIVE), False,
                             int(cv2.IMWRITE_JPEG_OPTIMIZE), True])
                        img_byte_arr = io.BytesIO(buffer)
                    else:
                        img = Image.open(img_stream)
                        rgb_im = img.convert('RGB')
                        rgb_im.thumbnail(PILLOW_THUMBNAIL_MAX_SIZE)
                        img_byte_arr = io.BytesIO()
                        rgb_im.save(img_byte_arr,
                                    format='JPEG',
                                    progressive=False,
                                    optimize=True,
                                    quality=85)
                    img_byte_arr.seek(0)
                    img_stream.close()
                    stream_to_compress = img_byte_arr.read()
                else:
                    # return base64.urlsafe_b64encode(bin_data)
                    stream_to_compress = bin_data

                if use_lzma:
                    return compressor.compress(stream_to_compress)
                else:
                    gzipFile.write(stream_to_compress)
                    gzipFile.close()
                    bytes_out.seek(0)
                    return bytes_out.read()

        return b''

    def pre_process_date_strings(str_in):
        # Split on the first new line, because usually the rest are footnotes
        str_in = str_in.split('\n')[0]
        # Quick fix for footnote '[1]'
        str_in = str_in.replace('[1]', '')

        # Some special cases
        if str_in.startswith('2D:'):
            str_in = str_in.replace('2D:', '')
        if str_in.startswith('3D:'):
            str_in = str_in.replace('3D:', '')
        str_splitted = str_in.split(': ')
        if len(str_splitted) > 1:
            str_in = str_splitted[1]
        str_splitted = str_in.split('(')
        if len(str_splitted) > 1:
            str_in = str_splitted[0]

        return str_in

    def map_dates(str_in, ignore_year=False):
        if str_in is None or str_in == '' or (isinstance(str_in, float) and math.isnan(str_in)):
            return None

        try:
            str_processed = pre_process_date_strings(str_in)
            dt = date_parser.parse(str_processed, ignoretz=True, fuzzy=True)
        except ParserError as e:
            # Deal with day first format
            if str(e).startswith('month must be'):
                try:
                    dt = date_parser.parse(str_processed, ignoretz=True, fuzzy=True, dayfirst=True)
                except ParserError as e:
                    print("Date parsing error. Input string: " + str_in)
                    print("Date parsing error. Input string processed: " + str_processed)
                    warnings.warn("Error: " + str(e))
                    return str_in
            else:
                print("Date parsing error. Input string: " + str_in)
                print("Date parsing error. Input string processed: " + str_processed)
                warnings.warn("Error: " + str(e))
                return str_in

        if ignore_year:
            return dt.strftime('%m-%d')
        else:
            return dt.strftime('%Y-%m-%d')

    df = pd.DataFrame.from_dict(data, orient='index')
    df_chara = pd.DataFrame(list(df['character']))
    df_app = df.iloc[:, :1]
    df_app['bookmarked'] = False

    # Map image filenames to actual data
    map_fn = functools.partial(map_image_file, base_path=img_base_path) if embed_image else lambda x: x
    df['profile_img'] = df['profile_img'].apply(map_fn)

    # Map dates
    df_chara['debut_date'] = df_chara['debut_date'].apply(map_dates)
    map_fn = functools.partial(map_dates, ignore_year=True)
    df_chara['birthday'] = df_chara['birthday'].apply(map_fn)

    # db file name
    if embed_image:
        database_path += '_embedded'
    database_path += '.db'

    # Remove previous db file
    if os.path.exists(database_path):
        os.remove(database_path)

    engine = create_engine('sqlite+pysqlite:///' + database_path,
                           echo=False, future=True)

    with engine.connect() as conn:
        metadata = MetaData()
        #     metadata.reflect(bind=engine)

        # Create tables
        columns = list(df.drop(['name', 'is_agency', 'character', 'ext_links'], axis=1).columns)
        columns = [Column(x, String) for x in columns]
        basic_info_table = Table(
            'basic_info',
            metadata,
            Column('name', String, primary_key=True, nullable=False),
            Column('is_agency', Boolean, nullable=False),
            *columns
        )

        # Create character table
        columns = sorted(set().union(*[x.keys() for x in df.character]))
        columns = [Column(x, String) for x in columns if x != 'name']
        character_table = Table(
            "character",
            metadata,
            Column('name', ForeignKey('basic_info.name'), nullable=False),
            *columns
        )

        # Create external link table
        columns = sorted(set().union(*[x.keys() for x in df.ext_links]))
        columns = [Column(x, String) for x in columns if x != 'name']
        ext_links_table = Table(
            "ext_links",
            metadata,
            Column('name', ForeignKey('basic_info.name'), nullable=False),
            *columns
        )

        if include_app_tables:
            # Create properties table for app
            columns = [Column('bookmarked', Boolean)]
            app_table = Table(
                "app",
                metadata,
                Column('name', ForeignKey('basic_info.name'), nullable=False),
                *columns
            )

        metadata.create_all(engine)
        # print(metadata.tables)

    df.drop(
        ['character', 'ext_links'], axis=1).to_sql(
        'basic_info', engine, index=False, if_exists='append')
    df_chara.to_sql(
        'character', engine, index=False, if_exists='append')
    pd.DataFrame(list(df['ext_links'])).to_sql(
        'ext_links', engine, index=False, if_exists='append')
    if include_app_tables:
        df_app.to_sql(
            'app', engine, index=False, if_exists='append')


if __name__ == '__main__':
    # with open('out/data.json', 'w', encoding='utf8') as f:
    #     json_str = parse_xml('data/virtualyoutuberfandomcom-20210712-wikidump', 'virtualyoutuberfandomcom-20210712')
    #     f.write(json_str)

    with open('out/data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
        json_to_sql(
            data,
            'out/wiki-data',
            'data/virtualyoutuberfandomcom-20210712-wikidump/images',
            embed_image=True,
            include_app_tables=False
        )

    print('Finished')
