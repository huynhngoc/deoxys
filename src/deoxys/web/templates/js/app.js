class App {
    constructor() {
        this.node_graph = document.getElementById('node_graph');
        this.activation_map = document.getElementById('activation_map');
        this.activation_map_btn = document.getElementById('btnActivationMap');
    }

    start() {
        $.get('/api/graph_nodes', {}, (data) => {
            console.log(this)
            const {
                edges,
                nodes
            } = data;

            const num = nodes.length
            console.log(num)

            const vis_data = {
                nodes: new vis.DataSet(nodes.map((value, index) => {
                    let x, y = 0;
                    if (index > num / 2) {
                        x = 50
                        y = (num - index) * 70
                    } else {
                        x = 10
                        y = index * 70
                    }

                    return {
                        x,
                        y,
                        mass: 1,
                        ...value
                    }
                })),
                edges: new vis.DataSet(edges.map(val => {
                    return { arrows: 'to', length: 30, ...val }
                }))
            };

            this.network = new vis.Network(this.node_graph, vis_data, {});
        });

        $(this.activation_map_btn).click(() => this.viewActivationMap())
    }

    viewActivationMap() {
        const selectedNodes = this.network.getSelectedNodes()

        if (selectedNodes.length == 1) {
            $(this.activation_map).empty()
            const layer_name = selectedNodes[0]
            $.get({
                url: '/api/activation_map',
                headers: {
                    layer_name,
                    img_id: 14
                }, success: (data) => {
                    data.forEach(element => {
                        $(this.activation_map).append(`<div class='d-block'>${element}</div>`)
                    });
                }
            })
        }
    }
}
