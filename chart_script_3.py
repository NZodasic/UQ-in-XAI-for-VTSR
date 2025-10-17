import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Create a comprehensive comparison table using Plotly
fig = go.Figure()

# Define the data for the three scenarios
scenarios = [
    {
        'row': 'Low Uncertainty',
        'input': 'Clear Stop Sign',
        'prediction': 'Stop Sign',
        'confidence': '0.95',
        'epistemic': '0.05',
        'aleatoric': '0.03', 
        'total': '0.06',
        'decision': '✓ ACCEPT',
        'gradcam': 'Good Focus',
        'explanation': 'Sign Shape & Color',
        'status': 'RELIABLE',
        'color': '#B3E5EC'
    },
    {
        'row': 'Med Uncertainty', 
        'input': 'Occluded Sign',
        'prediction': 'Speed Limit',
        'confidence': '0.72',
        'epistemic': '0.15',
        'aleatoric': '0.12',
        'total': '0.19', 
        'decision': '⚠ FLAG REVIEW',
        'gradcam': 'Mixed Focus',
        'explanation': 'Sign + Background',
        'status': 'UNCERTAIN',
        'color': '#FFEB8A'
    },
    {
        'row': 'High Uncertainty',
        'input': 'Blurry Sign',
        'prediction': 'Warning Sign', 
        'confidence': '0.48',
        'epistemic': '0.35',
        'aleatoric': '0.28',
        'total': '0.45',
        'decision': '✗ REJECT',
        'gradcam': 'Poor Focus',
        'explanation': 'Background Noise', 
        'status': 'UNRELIABLE',
        'color': '#FFCDD2'
    }
]

# Create table data
headers = ['Scenario', 'Input', 'Prediction', 'Conf', 'Epist', 'Aleat', 'Total', 'Decision', 'Grad-CAM', 'Status']

# Prepare table values with colors
values = []
colors = []

for scenario in scenarios:
    row_values = [
        scenario['row'],
        scenario['input'], 
        scenario['prediction'],
        scenario['confidence'],
        scenario['epistemic'],
        scenario['aleatoric'],
        scenario['total'],
        scenario['decision'],
        f"{scenario['gradcam']}<br>{scenario['explanation']}",
        scenario['status']
    ]
    values.append(row_values)
    colors.append([scenario['color']] * len(headers))

# Transpose for table format
table_values = list(map(list, zip(*values)))
table_colors = list(map(list, zip(*colors)))

# Create the table
fig.add_trace(go.Table(
    header=dict(
        values=headers,
        fill_color='#1FB8CD',
        font=dict(color='white', size=12),
        align='center',
        height=40
    ),
    cells=dict(
        values=table_values,
        fill_color=table_colors,
        font=dict(color='black', size=11),
        align='center',
        height=60
    )
))

# Add legend information as annotations
fig.add_annotation(
    text="Legend: 🟢 Low < 0.15 (Reliable) | 🟡 Med 0.15-0.30 (Caution) | 🔴 High > 0.30 (Unreliable)",
    xref="paper", yref="paper",
    x=0.5, y=-0.1,
    showarrow=False,
    font=dict(size=12),
    xanchor='center'
)

fig.add_annotation(
    text="UQ & XAI work together for reliable predictions",
    xref="paper", yref="paper", 
    x=0.5, y=-0.15,
    showarrow=False,
    font=dict(size=10, style='italic'),
    xanchor='center'
)

fig.update_layout(
    title="UQ & XAI Prediction Analysis",
    title_x=0.5,
    showlegend=False
)

# Save the chart
fig.write_image("uncertainty_gradcam_comparison.png")
fig.write_image("uncertainty_gradcam_comparison.svg", format="svg")

print("Chart saved successfully as PNG and SVG")