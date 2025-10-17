import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Create figure with custom layout
fig = go.Figure()

# Set up the canvas
fig.update_layout(
    title="Explainable AI (XAI): Grad-CAM for Traffic Sign Recognition",
    showlegend=False,
    xaxis=dict(range=[0, 10], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[0, 12], showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white'
)

# Top Section: What is Explainable AI
fig.add_shape(type="rect", x0=0.5, y0=10.5, x1=9.5, y1=11.8,
              fillcolor="#B3E5EC", line=dict(color="#1FB8CD", width=2))
fig.add_annotation(x=5, y=11.6, text="<b>What is Explainable AI (XAI)?</b>", 
                   font=dict(size=16, color="#13343B"), showarrow=False)
fig.add_annotation(x=5, y=11.2, text="Why did the model predict this traffic sign class?", 
                   font=dict(size=12, color="#13343B"), showarrow=False)
fig.add_annotation(x=5, y=10.8, text="Purpose: Visual explanations • Importance: Trust, Safety, Debugging", 
                   font=dict(size=10, color="#13343B"), showarrow=False)

# Grad-CAM Process - 5 steps horizontally
process_y = 8.5
step_width = 1.6
step_height = 1.2
colors = ["#A5D6A7", "#A5D6A7", "#A5D6A7", "#A5D6A7", "#A5D6A7"]

# Step boxes
for i in range(5):
    x_start = 0.5 + i * 1.8
    x_end = x_start + step_width
    
    fig.add_shape(type="rect", x0=x_start, y0=process_y-0.6, x1=x_end, y1=process_y+0.6,
                  fillcolor=colors[i], line=dict(color="#2E8B57", width=2))
    
    # Add arrows between steps
    if i < 4:
        fig.add_annotation(x=x_end + 0.1, y=process_y, text="→", 
                          font=dict(size=20, color="#2E8B57"), showarrow=False)

# Step 1: Input Image
fig.add_annotation(x=1.3, y=process_y+0.3, text="<b>Step 1: Input</b>", 
                   font=dict(size=10, color="#13343B"), showarrow=False)
fig.add_annotation(x=1.3, y=process_y, text="Traffic sign image", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=1.3, y=process_y-0.3, text="Original Input", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Step 2: Forward Pass
fig.add_annotation(x=3.1, y=process_y+0.3, text="<b>Step 2: Forward</b>", 
                   font=dict(size=10, color="#13343B"), showarrow=False)
fig.add_annotation(x=3.1, y=process_y, text="CNN processes", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=3.1, y=process_y-0.3, text="Get predictions", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Step 3: Compute Gradients
fig.add_annotation(x=4.9, y=process_y+0.3, text="<b>Step 3: Gradients</b>", 
                   font=dict(size=10, color="#13343B"), showarrow=False)
fig.add_annotation(x=4.9, y=process_y, text="Backpropagate", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=4.9, y=process_y-0.3, text="∂y^c/∂A", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Step 4: Weight Activations
fig.add_annotation(x=6.7, y=process_y+0.3, text="<b>Step 4: Weight</b>", 
                   font=dict(size=10, color="#13343B"), showarrow=False)
fig.add_annotation(x=6.7, y=process_y, text="α_k = mean(grad)", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=6.7, y=process_y-0.3, text="L = Σ(α_k × A_k)", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Step 5: Generate Heatmap
fig.add_annotation(x=8.5, y=process_y+0.3, text="<b>Step 5: Heatmap</b>", 
                   font=dict(size=10, color="#13343B"), showarrow=False)
fig.add_annotation(x=8.5, y=process_y, text="Apply ReLU", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=8.5, y=process_y-0.3, text="Overlay on image", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Interpretation Section - 3 boxes side by side
interp_y = 5.5
box_width = 2.8

# Red/Hot regions
fig.add_shape(type="rect", x0=0.5, y0=interp_y-0.8, x1=0.5+box_width, y1=interp_y+0.8,
              fillcolor="#FFCDD2", line=dict(color="#DB4545", width=2))
fig.add_annotation(x=1.9, y=interp_y+0.5, text="<b>Red/Hot Regions</b>", 
                   font=dict(size=11, color="#13343B"), showarrow=False)
fig.add_annotation(x=1.9, y=interp_y+0.1, text="Most important", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=1.9, y=interp_y-0.2, text="Model focuses here", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=1.9, y=interp_y-0.5, text="Sign features", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Yellow/Warm regions
fig.add_shape(type="rect", x0=3.6, y0=interp_y-0.8, x1=3.6+box_width, y1=interp_y+0.8,
              fillcolor="#FFEB8A", line=dict(color="#D2BA4C", width=2))
fig.add_annotation(x=5.0, y=interp_y+0.5, text="<b>Yellow/Warm</b>", 
                   font=dict(size=11, color="#13343B"), showarrow=False)
fig.add_annotation(x=5.0, y=interp_y+0.1, text="Moderately important", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=5.0, y=interp_y-0.2, text="Supporting evidence", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=5.0, y=interp_y-0.5, text="Context info", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Blue/Cold regions
fig.add_shape(type="rect", x0=6.7, y0=interp_y-0.8, x1=6.7+box_width, y1=interp_y+0.8,
              fillcolor="#B3E5EC", line=dict(color="#1FB8CD", width=2))
fig.add_annotation(x=8.1, y=interp_y+0.5, text="<b>Blue/Cold</b>", 
                   font=dict(size=11, color="#13343B"), showarrow=False)
fig.add_annotation(x=8.1, y=interp_y+0.1, text="Least important", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=8.1, y=interp_y-0.2, text="Background", 
                   font=dict(size=9, color="#13343B"), showarrow=False)
fig.add_annotation(x=8.1, y=interp_y-0.5, text="Ignored by model", 
                   font=dict(size=9, color="#13343B"), showarrow=False)

# Key Insight Box
fig.add_shape(type="rect", x0=1.5, y0=2.2, x1=8.5, y1=3.5,
              fillcolor="#9FA8B0", line=dict(color="#5D878F", width=2))
fig.add_annotation(x=5, y=3.2, text="<b>Key Insight</b>", 
                   font=dict(size=14, color="#13343B"), showarrow=False)
fig.add_annotation(x=5, y=2.9, text="Good XAI: Focuses on sign shape, color, symbols", 
                   font=dict(size=11, color="#13343B"), showarrow=False)
fig.add_annotation(x=5, y=2.5, text="Bad XAI: Focuses on background, spurious correlations", 
                   font=dict(size=11, color="#13343B"), showarrow=False)

# Add section headers
fig.add_annotation(x=5, y=9.5, text="<b>Grad-CAM Process</b>", 
                   font=dict(size=16, color="#2E8B57"), showarrow=False)
fig.add_annotation(x=5, y=6.8, text="<b>Interpretation</b>", 
                   font=dict(size=16, color="#D2BA4C"), showarrow=False)

# Add heatmap gradient visualization
gradient_x = [7.5, 8.0, 8.5, 9.0]
gradient_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']  # Blue to Red
for i, (x, color) in enumerate(zip(gradient_x, gradient_colors)):
    fig.add_shape(type="rect", x0=x-0.15, y0=1.2, x1=x+0.15, y1=1.6,
                  fillcolor=color, line=dict(width=0))

fig.add_annotation(x=8.25, y=1.0, text="Heatmap Colors", 
                   font=dict(size=10, color="#13343B"), showarrow=False)
fig.add_annotation(x=7.5, y=0.7, text="Cold", 
                   font=dict(size=8, color="#13343B"), showarrow=False)
fig.add_annotation(x=9.0, y=0.7, text="Hot", 
                   font=dict(size=8, color="#13343B"), showarrow=False)

# Save the figure
fig.write_image("grad_cam_xai_diagram.png")
fig.write_image("grad_cam_xai_diagram.svg", format="svg")

print("Grad-CAM XAI diagram created successfully!")