from _1_config import *
from _3_utils import *

def plot_predictions(plot_data, title):
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(plot_data['dates']['train'], plot_data['values']['train'], 
             label='Train', color='lightgray')
    
    # Plot test data and predictions
    plt.plot(plot_data['dates']['test'], plot_data['values']['test'], 
             label='Test', color='blue')
    plt.plot(plot_data['dates']['test'], plot_data['values']['test_predictions'], 
             label='Test Predictions', linestyle='--', color='green')
    
    # Plot future predictions
    plt.plot(plot_data['dates']['future'], plot_data['values']['future_predictions'], 
             label='Future Predictions', linestyle='--', color='red')

    # Add markers for highest and lowest points
    test_high_idx = np.argmax(plot_data['values']['test'])
    test_low_idx = np.argmin(plot_data['values']['test'])
    pred_high_idx = np.argmax(plot_data['values']['test_predictions'])
    pred_low_idx = np.argmin(plot_data['values']['test_predictions'])

    # Add markers
    plt.scatter(plot_data['dates']['test'][test_high_idx], 
               plot_data['values']['test'][test_high_idx], 
               color='green', marker='^', s=100, label='Highest Test Close')
    plt.scatter(plot_data['dates']['test'][test_low_idx], 
               plot_data['values']['test'][test_low_idx], 
               color='red', marker='v', s=100, label='Lowest Test Close')
    plt.scatter(plot_data['dates']['test'][pred_high_idx], 
               plot_data['values']['test_predictions'][pred_high_idx], 
               color='blue', marker='^', s=100, label='Highest Predicted Close')
    plt.scatter(plot_data['dates']['test'][pred_low_idx], 
               plot_data['values']['test_predictions'][pred_low_idx], 
               color='orange', marker='v', s=100, label='Lowest Predicted Close')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
