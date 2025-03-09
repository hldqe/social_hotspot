// 全局变量
let networkData = null;
let networkGraph = null;

// 页面加载后执行
document.addEventListener('DOMContentLoaded', function() {
    // 预处理数据
    fetch('/preprocess')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateNetworkStats(data.stats);
                // 加载网络数据
                loadNetworkData();
            } else {
                alert('数据预处理失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('预处理请求错误:', error);
            alert('数据预处理请求失败，请检查控制台获取详细信息');
        });
});

// 加载网络数据
function loadNetworkData() {
    fetch('/get_network_data')
        .then(response => response.json())
        .then(data => {
            networkData = data;
            console.log('网络数据加载成功', networkData);
            
            // 更新网络统计信息
            updateNetworkStats(networkData.stats);
            
            // 创建网络图
            createNetworkGraph();
            
            // 更新节点表格
            updateNodeTable();
        })
        .catch(error => {
            console.error('获取网络数据失败:', error);
            alert('获取网络数据失败，请检查控制台获取详细信息');
        });
}

// 创建网络图
function createNetworkGraph() {
    // 准备节点数据
    const nodes = networkData.nodes;
    const edges = networkData.edges;
    
    // 创建节点轨迹
    const nodeTrace = {
        x: [], y: [], 
        mode: 'markers',
        hoverinfo: 'text',
        text: [],
        marker: {
            color: [],
            size: [],
            line: {
                width: 1,
                color: 'black'
            }
        },
        ids: []
    };
    
    // 使用力导向算法计算节点位置
    const positions = calculateNodePositions(nodes, edges);
    
    // 设置节点数据
    nodes.forEach((node, i) => {
        const pos = positions[node.id] || { x: Math.random(), y: Math.random() };
        
        nodeTrace.x.push(pos.x);
        nodeTrace.y.push(pos.y);
        
        // 节点ID和悬停文本
        nodeTrace.ids.push(node.id);
        nodeTrace.text.push(`节点ID: ${node.id}<br>热点概率: ${node.prediction.toFixed(2)}<br>连接数: ${node.degree}<br>中介中心性: ${node.betweenness.toFixed(4)}`);
        
        // 节点颜色和大小
        nodeTrace.marker.color.push(node.prediction > 0.5 ? 'red' : 'blue');
        nodeTrace.marker.size.push(10 + Math.sqrt(node.degree) * 3);
    });
    
    // 创建边轨迹
    const edgeTrace = {
        x: [], y: [],
        mode: 'lines',
        line: {
            width: 1,
            color: 'rgba(150,150,150,0.3)'
        },
        hoverinfo: 'none'
    };
    
    // 设置边数据
    edges.forEach(edge => {
        const sourcePos = positions[edge.source] || { x: 0, y: 0 };
        const targetPos = positions[edge.target] || { x: 0, y: 0 };
        
        edgeTrace.x.push(sourcePos.x, targetPos.x, null);
        edgeTrace.y.push(sourcePos.y, targetPos.y, null);
    });
    
    // 创建Plotly图形
    const data = [edgeTrace, nodeTrace];
    const layout = {
        title: { text: '社交网络热点节点可视化', font: { size: 16 } },
        showlegend: false,
        hovermode: 'closest',
        margin: { b: 20, l: 20, r: 20, t: 40 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        dragmode: 'pan',
        annotations: [
            { text: "热点节点", x: 0.01, y: 0.99, showarrow: false, font: { color: 'red' } },
            { text: "非热点节点", x: 0.01, y: 0.95, showarrow: false, font: { color: 'blue' } }
        ]
    };
    
    Plotly.newPlot('networkGraph', data, layout, { responsive: true });
    
    // 添加点击事件
    document.getElementById('networkGraph').on('plotly_click', function(eventData) {
        const pointIndex = eventData.points[0].pointIndex;
        const nodeId = nodeTrace.ids[pointIndex];
        const node = networkData.nodes.find(n => n.id === nodeId);
        
        if (node) {
            displayNodeDetails(node);
            highlightNodeInTable(node.id);
        }
    });
    
    // 添加"重置视图"按钮事件
    document.getElementById('resetView').addEventListener('click', function() {
        Plotly.relayout('networkGraph', {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        });
    });
}

// 使用简单的力导向算法计算节点位置
function calculateNodePositions(nodes, edges) {
    // 创建一个简单的位置映射
    const positions = {};
    
    // 初始化随机位置
    nodes.forEach(node => {
        positions[node.id] = {
            x: Math.random() * 2 - 1,
            y: Math.random() * 2 - 1
        };
    });
    
    // 简单的力导向算法 (实际应用中应使用更复杂的算法)
    const iterations = 50;
    const k = 0.1; // 弹簧常数
    
    for (let iter = 0; iter < iterations; iter++) {
        // 计算斥力 (节点间)
        for (let i = 0; i < nodes.length; i++) {
            for (let j = 0; j < nodes.length; j++) {
                if (i === j) continue;
                
                const nodeA = nodes[i];
                const nodeB = nodes[j];
                const posA = positions[nodeA.id];
                const posB = positions[nodeB.id];
                
                const dx = posA.x - posB.x;
                const dy = posA.y - posB.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
                
                // 斥力
                const force = k / dist;
                posA.x += dx * force * 0.05;
                posA.y += dy * force * 0.05;
            }
        }
        
        // 计算引力 (连接的节点间)
        for (const edge of edges) {
            const posA = positions[edge.source];
            const posB = positions[edge.target];
            
            if (!posA || !posB) continue;
            
            const dx = posA.x - posB.x;
            const dy = posA.y - posB.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
            
            // 引力
            const force = dist * k;
            posA.x -= dx * force * 0.05;
            posA.y -= dy * force * 0.05;
            posB.x += dx * force * 0.05;
            posB.y += dy * force * 0.05;
        }
    }
    
    return positions;
}

// 更新网络统计信息
function updateNetworkStats(stats) {
    const statsElement = document.getElementById('networkStats');
    if (statsElement && stats) {
        statsElement.innerHTML = `
            <p><strong>节点数量:</strong> ${stats.nodeCount}</p>
            <p><strong>连接数量:</strong> ${stats.edgeCount}</p>
            <p><strong>热点节点数:</strong> ${stats.hotspotCount}</p>
            <p><strong>热点占比:</strong> ${(stats.hotspotCount / stats.nodeCount * 100).toFixed(2)}%</p>
        `;
    }
}

// 显示节点详情
function displayNodeDetails(node) {
    const detailsElement = document.getElementById('nodeDetails');
    if (detailsElement && node) {
        const isHotspot = node.prediction > parseFloat(document.getElementById('thresholdValue').textContent);
        
        detailsElement.innerHTML = `
            <h5>节点 ${node.id}</h5>
            <p><strong>热点概率:</strong> ${node.prediction.toFixed(4)}</p>
            <p><strong>连接数:</strong> ${node.degree}</p>
            <p><strong>中介中心性:</strong> ${node.betweenness.toFixed(4)}</p>
            <p><strong>判定结果:</strong> 
                <span class="badge ${isHotspot ? 'badge-hotspot' : 'badge-normal'}">
                    ${isHotspot ? '热点' : '非热点'}
                </span>
            </p>
            <p><strong>真实标签:</strong> 
                <span class="badge ${node.label == 1 ? 'badge-hotspot' : 'badge-normal'}">
                    ${node.label == 1 ? '热点' : '非热点'}
                </span>
            </p>
        `;
    }
}

// 更新节点表格
function updateNodeTable() {
    const tableBody = document.getElementById('nodeTableBody');
    if (!tableBody || !networkData) return;
    
    // 获取当前阈值
    const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
    
    // 清空表格
    tableBody.innerHTML = '';
    
    // 按热点概率排序
    const sortedNodes = [...networkData.nodes].sort((a, b) => b.prediction - a.prediction);
    
    // 更新表格
    sortedNodes.forEach(node => {
        const isHotspot = node.prediction > threshold;
        
        const row = tableBody.insertRow();
        row.id = `node-row-${node.id}`;
        row.innerHTML = `
            <td>${node.id}</td>
            <td>${node.prediction.toFixed(4)}</td>
            <td>${node.degree}</td>
            <td>${node.betweenness.toFixed(4)}</td>
            <td><span class="badge ${isHotspot ? 'badge-hotspot' : 'badge-normal'}">${isHotspot ? '热点' : '非热点'}</span></td>
        `;
        
        // 添加点击事件
        row.addEventListener('click', function() {
            displayNodeDetails(node);
            // 也可以在网络图中高亮显示该节点
        });
    });
}

// 高亮表格中的节点
function highlightNodeInTable(nodeId) {
    // 移除所有高亮
    document.querySelectorAll('#nodeTableBody tr').forEach(row => {
        row.classList.remove('node-highlighted');
    });
    
    // 添加高亮
    const row = document.getElementById(`node-row-${nodeId}`);
    if (row) {
        row.classList.add('node-highlighted');
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}
