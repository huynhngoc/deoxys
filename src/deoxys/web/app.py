# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from sanic import Sanic
from sanic.response import file, json, html
import pkgutil
import os
from ..experiment import MultiExperimentDB
from ..utils import write_byte, read_file


class BaseApp:
    def __init__(self):
        self.app = Sanic()

        # @self.app.route("/")
        # async def test(request):
        #     return json({"hello": "world"})

    def add_route_handler(self, uri, handler, *args, **kwargs):
        self.app.route(uri, *args, **kwargs)(handler)

    def run(self):
        self.app.run(host="0.0.0.0", port=8000)


class VisApp(BaseApp):
    def __init__(self, dbclient, basepath='.'):
        super().__init__()
        self.experiments = MultiExperimentDB(dbclient)
        self.basepath = basepath

        self._generate_static_assets()
        self._map_static_files()

        self.create_home_page()

    def create_home_page(self):
        html_content = read_file(os.path.join(
            self.basepath, 'assets/html/index.html'))

        @self.app.route('/')
        async def handler(resquest):
            return html(html_content)

    def _generate_static_assets(self):
        basepath = self.basepath

        html_path = os.path.join(basepath, 'assets/html')
        js_path = os.path.join(basepath, 'assets/js')
        css_path = os.path.join(basepath, 'assets/css')

        if not os.path.exists(basepath):
            os.makedirs(basepath)

        if not os.path.exists(html_path):
            os.makedirs(html_path)

        if not os.path.exists(js_path):
            os.makedirs(js_path)

        if not os.path.exists(css_path):
            os.makedirs(css_path)

        html_file = pkgutil.get_data(__name__, "templates/html/index.html")
        write_byte(html_file, os.path.join(html_path, 'index.html'))

        js_file = pkgutil.get_data(__name__, "templates/js/app.js")
        write_byte(js_file, os.path.join(js_path, 'app.js'))

        css_file = pkgutil.get_data(__name__, "templates/css/app.css")
        write_byte(css_file, os.path.join(css_path, 'app.css'))

    def _map_static_files(self):
        self.app.static('/static', os.path.join(self.basepath,
                                                'assets'), name='static')
        self.app.url_for('static', filename='any')
