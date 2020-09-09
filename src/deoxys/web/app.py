# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from sanic import Sanic
from sanic.response import file, json, html
import pkgutil
import os
import matplotlib.pyplot as plt
import mpld3
from ..experiment import MultiExperimentDB
from ..utils import write_byte, read_file
from ..model import load_model


class BaseApp:
    def __init__(self):
        self.app = Sanic()

        # @self.app.route("/")
        # async def test(request):
        #     return json({"hello": "world"})
        self.working_model = load_model(
            # '../../hn_perf/logs_db/'
            # '5e3547f57a4c104db6d4c5d2/model/model.004.h5'
            '../../oxford_perf/logs_db/'
            '5e2e2349a356a4893813c8f7/model/model.022.h5'
        )

    def add_route_handler(self, uri, handler, *args, **kwargs):
        self.app.route(uri, *args, **kwargs)(handler)

    def run(self):
        self.app.run(host="0.0.0.0", port=8000)


class VisApp(BaseApp):
    def __init__(self, dbclient, basepath='.', provider='mongodb'):
        super().__init__()
        self.me = MultiExperimentDB(dbclient)
        self.basepath = basepath

        if 'env' in os.environ and os.environ['ENV'] == 'dev':
            self._initialize_dev_mode()
        else:
            self._generate_static_assets()
            self._map_static_files()

            self.create_home_page()
        self.add_routes()

    def create_home_page(self):
        html_content = read_file(os.path.join(
            self.basepath, 'assets/html/index.html'))

        @self.app.route('/')
        async def handler(resquest):
            return html(html_content)

    def add_routes(self):
        @self.app.route('/api/experiments')
        async def all_experiment(request):
            exps = self.me.experiments

            data = self.me.json_data(exps)

            return json(data)

        @self.app.route('/api/sessions')
        async def all_sessions(request):
            headers = request.headers
            exp_id = headers['experiment_id']

            sessions = self.me.sessions_from_experiments(exp_id)

            data = self.me.json_data(sessions)

            return json(data)

        @self.app.route('/api/graph_nodes')
        def graph_nodes(request):
            # headers = request.headers
            # exp_id = headers['experiment_id']
            graph_nodes = self.working_model.node_graph
            layers = self.working_model.layers
            nodes = [{'id': layer,
                      'label': layer + '\n' + str(layers[layer].output.shape)}
                     for layer in layers
                     if 'resize' not in layer and 'concat' not in layer]

            return json({
                'nodes': nodes,
                'edges': graph_nodes
            })

        @self.app.route('/api/activation_map')
        def activation_map(request):
            headers = request.headers
            layer = headers['layer_name']
            img_id = int(headers['img_id'])

            model = self.working_model
            dr = model.data_reader

            batch_size = dr.batch_size
            batch_index = img_id // batch_size
            batch_pos = img_id % batch_size
            print(batch_index, batch_pos)
            for i, img in enumerate(dr.val_generator.generate()):
                if i == batch_index:
                    image = img[0][batch_pos:batch_pos + 1]
                    break

            print(image.shape)
            print('getting activation map')
            activation_map = model.activation_map(layer, image)[0]

            print('plotting')
            filter_num = activation_map.shape[-1]
            filters = []
            img_size = activation_map.shape[:-1]
            figsize = (img_size[0] / 50, img_size[1] / 50)
            if figsize[0] > 3 or figsize[1] > 3:
                figsize = (3, 3 * img_size[1] / img_size[0])
            elif figsize[0] < 2 or figsize[1] < 2:
                figsize = (1, 1 * img_size[1] / img_size[0])
            for i in range(filter_num):
                fig = plt.figure(figsize=figsize)
                # if image.shape[-1] == 3:
                #     plt.imshow(image[0])
                # else:
                #     plt.imshow(image[0][..., 0])
                plt.imshow(activation_map[..., i], alpha=0.3, cmap='gray')
                mpld3.plugins.connect(fig, mpld3.plugins.BoxZoom(False, False))
                filters.append(mpld3.fig_to_html(fig))
                plt.close('all')

            return json(filters)

    def _initialize_dev_mode(self):
        @self.app.route('/')
        async def handler(resquest):
            html_file = pkgutil.get_data(__name__, "templates/html/index.html")

            return html(str(html_file, 'utf-8'))

        self.app.static('/static/js',
                        os.path.join(os.path.dirname(__file__),
                                     "templates/js"), name='static/js')
        self.app.static('/static/css',
                        os.path.join(os.path.dirname(__file__),
                                     "templates/css"), name='static/css')
        self.app.static('/static/html',
                        os.path.join(os.path.dirname(__file__),
                                     "templates/html"), name='static')
        self.app.url_for('static', filename='any')

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
