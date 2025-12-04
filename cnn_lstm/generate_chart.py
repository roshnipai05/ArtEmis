import matplotlib.pyplot as plt
import os

# Data provided from your logs (Epochs 11-30)
# We will prepend dummy data for epochs 1-10 to visualize the trend, 
# assuming 1-10 were higher.
epochs = list(range(11, 31))

train_loss = [
    4.9336, 3.4973, 3.4624, 3.4285, 3.3975, 
    3.3673, 3.3371, 3.3094, 3.2819, 3.2538,
    3.2275, 3.2011, 3.1748, 3.1512, 3.1244,
    3.0724, 3.0572, 3.0425, 3.0141, 3.0059
]

val_loss = [
    4.4080, 3.6458, 3.6242, 3.6148, 3.6092, 
    3.5998, 3.5951, 3.5950, 3.5884, 3.5946,
    3.5891, 3.5876, 3.5909, 3.5949, 3.5941,
    3.5945, 3.5974, 3.5989, 3.5988, 3.6015
]

def create_loss_graph():
    plt.figure(figsize=(10, 6))
    
    # Plotting lines
    plt.plot(epochs, train_loss, label='Training Loss', color='#2ecc71', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_loss, label='Validation Loss', color='#e74c3c', linewidth=2, marker='s', markersize=4)
    
    # Styling
    plt.title('ArtEmis Training: SimpleCNN + LSTM', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Highlight the gap
    plt.annotate('Overfitting Gap', xy=(29, 3.05), xytext=(25, 3.4),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    
    # Save
    output_path = 'training_loss_graph.png'
    plt.savefig(output_path)
    print(f"Graph saved to {os.path.abspath(output_path)}")
    plt.show()

if __name__ == "__main__":
    create_loss_graph()