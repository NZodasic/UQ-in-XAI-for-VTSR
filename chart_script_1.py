import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Create a diagram showing Uncertainty Quantification concept
fig = go.Figure()

# Define positions for the boxes
# Section 1: Types of Uncertainty (top row)
epistemic_x = [0.05, 0.45, 0.45, 0.05, 0.05]
epistemic_y = [0.7, 0.7, 0.95, 0.95, 0.7]

aleatoric_x = [0.55, 0.95, 0.95, 0.55, 0.55]
aleatoric_y = [0.7, 0.7, 0.95, 0.95, 0.7]

# Section 2: Quantification Methods (middle row)
mc_x = [0.05, 0.45, 0.45, 0.05, 0.05]
mc_y = [0.4, 0.4, 0.65, 0.65, 0.4]

tta_x = [0.55, 0.95, 0.95, 0.55, 0.55]
tta_y = [0.4, 0.4, 0.65, 0.65, 0.4]

# Section 3: Combined Total Uncertainty (bottom row)
total_x = [0.2, 0.8, 0.8, 0.2, 0.2]
total_y = [0.05, 0.05, 0.35, 0.35, 0.05]

# Add boxes with updated colors
# Epistemic Uncertainty box (Blue)
fig.add_trace(go.Scatter(
    x=epistemic_x, y=epistemic_y,
    fill='toself',
    fillcolor='#B3E5EC',
    line=dict(color='#1FB8CD', width=3),
    mode='lines',
    name='Epistemic',
    showlegend=False
))

# Aleatoric Uncertainty box (Orange)
fig.add_trace(go.Scatter(
    x=aleatoric_x, y=aleatoric_y,
    fill='toself',
    fillcolor='#FFEB8A',
    line=dict(color='#D2BA4C', width=3),
    mode='lines',
    name='Aleatoric',
    showlegend=False
))

# Monte Carlo Dropout box (Blue)
fig.add_trace(go.Scatter(
    x=mc_x, y=mc_y,
    fill='toself',
    fillcolor='#B3E5EC',
    line=dict(color='#1FB8CD', width=3),
    mode='lines',
    name='Monte Carlo',
    showlegend=False
))

# Test-Time Augmentation box (Orange)
fig.add_trace(go.Scatter(
    x=tta_x, y=tta_y,
    fill='toself',
    fillcolor='#FFEB8A',
    line=dict(color='#D2BA4C', width=3),
    mode='lines',
    name='TTA',
    showlegend=False
))

# Total Uncertainty box (Green)
fig.add_trace(go.Scatter(
    x=total_x, y=total_y,
    fill='toself',
    fillcolor='#A5D6A7',
    line=dict(color='#2E8B57', width=3),
    mode='lines',
    name='Total',
    showlegend=False
))

# Add arrows with better positioning
# Arrow from Epistemic to Monte Carlo
fig.add_annotation(
    x=0.25, y=0.65, ax=0.25, ay=0.7,
    arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#333333",
    showarrow=True
)

# Arrow from Aleatoric to TTA
fig.add_annotation(
    x=0.75, y=0.65, ax=0.75, ay=0.7,
    arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#333333",
    showarrow=True
)

# Arrow from Monte Carlo to Total
fig.add_annotation(
    x=0.35, y=0.35, ax=0.25, ay=0.4,
    arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#333333",
    showarrow=True
)

# Arrow from TTA to Total
fig.add_annotation(
    x=0.65, y=0.35, ax=0.75, ay=0.4,
    arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#333333",
    showarrow=True
)

# Add text annotations with improved formatting and sizing
fig.add_annotation(
    x=0.25, y=0.825, 
    text="<b style='font-size:14px'>Epistemic Uncertainty</b><br><b style='font-size:12px'>(Model Uncertainty)</b><br><br><i style='font-size:11px'>What model doesn't know</i><br><br>• Reducible with more data<br>• Insufficient training data<br>• Model limitations<br><br><b style='color:#DB4545'>Example:</b> Unseen traffic sign",
    showarrow=False, font=dict(size=11), align='center'
)

fig.add_annotation(
    x=0.75, y=0.825, 
    text="<b style='font-size:14px'>Aleatoric Uncertainty</b><br><b style='font-size:12px'>(Data Uncertainty)</b><br><br><i style='font-size:11px'>Inherent noise in data</i><br><br>• Cannot be reduced<br>• Blur, occlusion<br>• Poor lighting, sensor noise<br><br><b style='color:#DB4545'>Example:</b> Blurry sign in fog",
    showarrow=False, font=dict(size=11), align='center'
)

fig.add_annotation(
    x=0.25, y=0.525, 
    text="<b style='font-size:14px'>Monte Carlo Dropout</b><br><br>• For Epistemic Uncertainty<br>• Enable dropout at test time<br>• Multiple forward passes (N=20-30)<br>• Mean prediction + variance<br><br><b style='color:#2E8B57'>Formula:</b> Epistemic = Std(predictions)",
    showarrow=False, font=dict(size=11), align='center'
)

fig.add_annotation(
    x=0.75, y=0.525, 
    text="<b style='font-size:14px'>Test-Time Augmentation</b><br><br>• For Aleatoric Uncertainty<br>• Apply random augmentations<br>• Multiple predictions<br>• Mean prediction + variance<br><br><b style='color:#2E8B57'>Formula:</b> Aleatoric = Std(augmented_preds)",
    showarrow=False, font=dict(size=11), align='center'
)

fig.add_annotation(
    x=0.5, y=0.2, 
    text="<b style='font-size:16px'>Total Uncertainty</b><br><br><b style='font-size:14px; color:#2E8B57'>Formula:</b> Total = √(Epistemic² + Aleatoric²)<br><br><b style='font-size:12px'>Interpretation:</b><br>High uncertainty → Low confidence → Defer to human<br><br><b style='font-size:12px'>Application:</b> Safety-critical decisions in autonomous driving",
    showarrow=False, font=dict(size=12), align='center'
)

# Add section titles with better styling
fig.add_annotation(
    x=0.5, y=0.98, 
    text="<b style='font-size:16px'>Section 1: Types of Uncertainty</b>",
    showarrow=False, font=dict(size=16, color='#333333'), align='center'
)

fig.add_annotation(
    x=0.5, y=0.68, 
    text="<b style='font-size:16px'>Section 2: Quantification Methods</b>",
    showarrow=False, font=dict(size=16, color='#333333'), align='center'
)

fig.add_annotation(
    x=0.5, y=0.38, 
    text="<b style='font-size:16px'>Section 3: Combined Total Uncertainty</b>",
    showarrow=False, font=dict(size=16, color='#333333'), align='center'
)

# Update layout
fig.update_layout(
    title=dict(
        text="<b>Uncertainty Quantification Framework</b>",
        font=dict(size=20),
        x=0.5
    ),
    xaxis=dict(
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),
    yaxis=dict(
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("uncertainty_quantification.png")
fig.write_image("uncertainty_quantification.svg", format="svg")

print("Chart saved as uncertainty_quantification.png and uncertainty_quantification.svg")