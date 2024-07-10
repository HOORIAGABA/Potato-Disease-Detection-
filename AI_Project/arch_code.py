import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Function to add an arrowed annotation
def add_arrow(ax, text, xy, xytext):
    ax.annotate(text, xy=xy, xytext=xytext,
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

# Title
ax.set_title('Model Architecture for Potato Disease Detection', fontsize=16)

# Hide axes
ax.axis('off')

# Add rectangles and text
rects = [
    ('Data Collection\n(PlantVillage Dataset)', (0.1, 0.9), (0.1, 0.8)),
    ('Data Loading & Preprocessing\n(tf.data.Dataset, Augmentation)', (0.1, 0.75), (0.1, 0.65)),
    ('Dataset Partitioning\n(Training, Validation, Testing)', (0.1, 0.6), (0.1, 0.5)),
    ('Model Architecture\n(CNN with multiple Conv2D and MaxPooling layers)', (0.1, 0.45), (0.1, 0.35)),
    ('Model Compilation\n(Optimizer: Adam, Loss: SparseCategoricalCrossentropy)', (0.1, 0.3), (0.1, 0.2)),
    ('Model Training\n(Epochs: 50, Metrics: Accuracy)', (0.1, 0.15), (0.1, 0.05)),
    ('Model Evaluation\n(Accuracy: 0.9936, Loss: 0.0175)', (0.5, 0.15), (0.5, -1.2)),
]

for text, xy, xytext in rects:
    add_arrow(ax, text, xy, xytext)

# CNN Architecture Details
cnn_layers = [
    'Input Layer\n(256x256x3)',
    'Conv2D Layer\n(filters=32, kernel_size=3, activation=ReLU)',
    'MaxPooling2D Layer\n(pool_size=2)',
    'Conv2D Layer\n(filters=64, kernel_size=3, activation=ReLU)',
    'MaxPooling2D Layer\n(pool_size=2)',
    'Conv2D Layer\n(filters=64, kernel_size=3, activation=ReLU)',
    'MaxPooling2D Layer\n(pool_size=2)',
    'Conv2D Layer\n(filters=64, kernel_size=3, activation=ReLU)',
    'MaxPooling2D Layer\n(pool_size=2)',
    'Conv2D Layer\n(filters=64, kernel_size=3, activation=ReLU)',
    'MaxPooling2D Layer\n(pool_size=2)',
    'Conv2D Layer\n(filters=64, kernel_size=3, activation=ReLU)',
    'MaxPooling2D Layer\n(pool_size=2)',
    'Flatten Layer',
    'Dense Layer\n(units=64, activation=ReLU)',
    'Output Layer\n(units=3, activation=Softmax)'
]

# Position details for CNN Architecture
x_pos = 0.5
y_pos = 0.9
y_step = 0.07

for layer in cnn_layers:
    ax.text(x_pos, y_pos, layer, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='lightblue'))
    y_pos -= y_step

plt.show()
