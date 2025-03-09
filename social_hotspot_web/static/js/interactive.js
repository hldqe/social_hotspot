// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 添加阈值滑块事件监听
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    
    if (thresholdSlider && thresholdValue) {
        thresholdSlider.addEventListener('input', function() {
            const threshold = parseFloat(this.value);
            thresholdValue.textContent = threshold.toFixed(2);
            
            // 更新节点颜色
            updateNodeColors(threshold);
            
            // 更新节点表格
            updateNodeTable();
        });
    }
    
    // 添加排序功能
    const sortSelect = document.getElementById('sortSelect');
    const applySort = document.getElementById('applySort');
    
    if (sortSelect && applySort) {
        applySort.addEventListener('click', function() {
            const sortBy = sortSelect.value;
            sortNodes(sortBy);
        });
    }
});

// 更新节点颜色
function updateNodeColors(threshold) {
    if (!networkData) return;
    
    // 更新节点颜色
    const update = {
        'marker.color': networkData.nodes.map(node => 
            node.prediction > threshold ? 'red' : 'blue'
        )
    };
    
    Plotly.restyle('networkGraph', update, 1); // 1是节点轨迹的索引
}

// 排序节点
function sortNodes(sortBy) {
    if (!networkData) return;
    
    // 获取表格体
    const tableBody = document.getElementById('nodeTableBody');
    if (!tableBody) return;
    
    // 获取当前阈值
    const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
    
    // 根据选择的字段排序
    let sortedNodes;
    switch (sortBy) {
        case 'prediction':
            sortedNodes = [...networkData.nodes].sort((a, b) => b.prediction - a.prediction);
            break;
        case 'degree':
            sortedNodes = [...networkData.nodes].sort((a, b) => b.degree - a.degree);
            break;
        case 'betweenness':
            sortedNodes = [...networkData.nodes].sort((a, b) => b.betweenness - a.betweenness);
            break;
        default:
            sortedNodes = [...networkData.nodes];
    }
    
    // 清空表格
    tableBody.innerHTML = '';
    
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
