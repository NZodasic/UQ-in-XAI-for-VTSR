import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create a flowchart using Plotly graph visualization
# Define node positions for the flowchart layout
nodes = {
    'Vietnamese Traffic\nSigns Dataset': (1, 10),
    'EDA & Data\nQuality Checks': (1, 9),
    'Data Cleaning &\nPreprocessing': (1, 8),
    'ResNet18 + Custom Head\nDropout for UQ': (1, 7),
    'Model Training\nCross-Entropy + Adam': (1, 6),
    'Uncertainty\nQuantification': (1, 5),
    'Monte Carlo Dropout\nEpistemic': (0, 4),
    'Test-Time Aug\nAleatoric': (2, 4),
    'Combined Uncertainty\nTotal = √(Epis² + Alea²)': (1, 3),
    'Explainable AI\nGrad-CAM & Saliency': (1, 2),
    'Model Evaluation\nAccuracy & ECE': (1, 1),
    'Deployment\nDecision': (1, 0),
    'Accept': (0, -1),
    'Flag Review': (1, -1),
    'Reject': (2, -1)
}

# Define connections (edges) between nodes
edges = [
    ('Vietnamese Traffic\nSigns Dataset', 'EDA & Data\nQuality Checks'),
    ('EDA & Data\nQuality Checks', 'Data Cleaning &\nPreprocessing'),
    ('Data Cleaning &\nPreprocessing', 'ResNet18 + Custom Head\nDropout for UQ'),
    ('ResNet18 + Custom Head\nDropout for UQ', 'Model Training\nCross-Entropy + Adam'),
    ('Model Training\nCross-Entropy + Adam', 'Uncertainty\nQuantification'),
    ('Uncertainty\nQuantification', 'Monte Carlo Dropout\nEpistemic'),
    ('Uncertainty\nQuantification', 'Test-Time Aug\nAleatoric'),
    ('Monte Carlo Dropout\nEpistemic', 'Combined Uncertainty\nTotal = √(Epis² + Alea²)'),
    ('Test-Time Aug\nAleatoric', 'Combined Uncertainty\nTotal = √(Epis² + Alea²)'),
    ('Combined Uncertainty\nTotal = √(Epis² + Alea²)', 'Explainable AI\nGrad-CAM & Saliency'),
    ('Explainable AI\nGrad-CAM & Saliency', 'Model Evaluation\nAccuracy & ECE'),
    ('Model Evaluation\nAccuracy & ECE', 'Deployment\nDecision'),
    ('Deployment\nDecision', 'Accept'),
    ('Deployment\nDecision', 'Flag Review'),
    ('Deployment\nDecision', 'Reject')
]

# Define node categories for coloring
node_categories = {
    'Vietnamese Traffic\nSigns Dataset': 'data',
    'EDA & Data\nQuality Checks': 'data', 
    'Data Cleaning &\nPreprocessing': 'data',
    'ResNet18 + Custom Head\nDropout for UQ': 'model',
    'Model Training\nCross-Entropy + Adam': 'model',
    'Uncertainty\nQuantification': 'uncertainty',
    'Monte Carlo Dropout\nEpistemic': 'uncertainty',
    'Test-Time Aug\nAleatoric': 'uncertainty',
    'Combined Uncertainty\nTotal = √(Epis² + Alea²)': 'uncertainty',
    'Explainable AI\nGrad-CAM & Saliency': 'uncertainty',
    'Model Evaluation\nAccuracy & ECE': 'evaluation',
    'Deployment\nDecision': 'decision',
    'Accept': 'decision',
    'Flag Review': 'decision',
    'Reject': 'decision'
}

# Color mapping
color_map = {
    'data': '#B3E5EC',
    'model': '#A5D6A7', 
    'uncertainty': '#FFEB8A',
    'evaluation': '#E1BEE7',
    'decision': '#FFCDD2'
}

# Create edge traces
edge_traces = []
for edge in edges:
    x0, y0 = nodes[edge[0]]
    x1, y1 = nodes[edge[1]]
    edge_traces.append(go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode='lines',
        line=dict(width=2, color='#333333'),
        hoverinfo='none',
        showlegend=False
    ))

# Create node trace
node_x = [nodes[node][0] for node in nodes]
node_y = [nodes[node][1] for node in nodes]
node_colors = [color_map[node_categories[node]] for node in nodes]
node_text = list(nodes.keys())

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    marker=dict(
        size=60,
        color=node_colors,
        line=dict(width=2, color='#333333')
    ),
    text=node_text,
    textposition='middle center',
    textfont=dict(size=9, color='#133433'),
    hoverinfo='none',
    showlegend=False
)

# Create figure
fig = go.Figure()

# Add edge traces
for trace in edge_traces:
    fig.add_trace(trace)

# Add node trace
fig.add_trace(node_trace)

# Add legend manually
legend_items = [
    ('Data Processing', '#B3E5EC'),
    ('Model Stages', '#A5D6A7'),
    ('Uncertainty & XAI', '#FFEB8A'),
    ('Evaluation', '#E1BEE7'),
    ('Decision', '#FFCDD2')
]

for i, (label, color) in enumerate(legend_items):
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=15, color=color, line=dict(width=2, color='#333333')),
        name=label,
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title='UQ Pipeline for Vietnamese Traffic Sign Recognition',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.5, 2.5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-2, 11]
    ),
    plot_bgcolor='white',
    annotations=[
        dict(
            x=1,
            y=-1.7,
            text="Decision based on uncertainty level: Low→Accept, Medium→Review, High→Reject",
            showarrow=False,
            font=dict(size=10, color='#666666')
        )
    ]
)

# Save the chart
fig.write_image('uq_pipeline_flowchart.png')
fig.write_image('uq_pipeline_flowchart.svg', format='svg')

print("UQ Pipeline flowchart saved successfully as PNG and SVG")