
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
from io import BytesIO

def create_flow_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Define steps
    steps = [
        "Data Ingestion", "Validation", "Feature Eng.", "Training",
        "Evaluation", "Model Registration", "Serving", "Monitoring"
    ]
    
    # Coordinates
    x_start = 0.1
    y_start = 0.5
    gap = 0.11
    box_width = 0.09
    box_height = 0.15

    for i, step in enumerate(steps):
        x = x_start + i * gap
        
        # Color logic
        color = 'lightblue'
        if step in ["Training", "Evaluation"]: color = 'lightgreen'
        if step in ["Serving", "Monitoring"]: color = 'lightcoral'
        
        # Draw box
        rect = patches.FancyBboxPatch((x, y_start), box_width, box_height, boxstyle="round,pad=0.02", 
                                      linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + box_width/2, y_start + box_height/2, step, 
                ha='center', va='center', fontsize=9, wrap=True)
        
        # Add arrow
        if i < len(steps) - 1:
            ax.arrow(x + box_width + 0.005, y_start + box_height/2, 
                     gap - box_width - 0.01, 0, 
                     head_width=0.02, head_length=0.01, fc='k', ec='k')

    # Feedback Loop arrow
    # From Monitoring to Training (schematic)
    x_mon = x_start + (len(steps)-1) * gap + box_width/2
    x_train = x_start + 3 * gap + box_width/2
    
    # Draw curved arrow
    arrow = patches.FancyArrowPatch((x_mon, y_start), (x_train, y_start),
                                    connectionstyle="arc3,rad=-0.5", 
                                    arrowstyle="->", linestyle="--", color='purple', lw=2)
    ax.add_patch(arrow)
    ax.text((x_mon + x_train)/2, y_start - 0.15, "Feedback / Retraining", 
            ha='center', va='center', color='purple', fontsize=10)

    plt.title("MLOps End-to-End Pipeline", fontsize=14)
    plt.tight_layout()
    
    # Save as PNG
    output_path = "outputs/pipeline_flow.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    with open("outputs/pipeline_flow_base64.txt", "w") as f:
        f.write(img_str)
    print("Base64 string saved to outputs/pipeline_flow_base64.txt")

if __name__ == "__main__":
    create_flow_diagram()
