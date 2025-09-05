import seaborn as sns
import matplotlib.pyplot as plt

# Packages to save modules
import pickle

#Adapt focal loss function as it initially had a different name during model fit 
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
    eps = 1e-8
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))

# Set seaborn graphs
sns.set()

# Load in results history for model 
with open("og_mtb_cnn_results.pkl", "rb") as f:
    results = pickle.load(f)

# Plot four plots for CNN-BiLSTM performance metrics
plt.style.use('ggplot')
fig, ax = plt.subplots(2,2, figsize=[15,15])

# Plot Binary Accuracy 
ax[0][0].plot(results.history['val_binary_accuracy'],'b')
ax[0][0].set_ylabel("Categorical Accuracy Score")
ax[0][0].plot(results.history['binary_accuracy'],'r')
ax[0][0].legend(['Validation set','Training set'])

# Plot Precision score 
ax[0][1].plot(results.history['val_precision'],'b')
ax[0][1].plot(results.history['precision'],'r')
ax[0][1].set_ylabel("Precision Score")
ax[0][1].legend(['Validation set','Training set'])

# Plot Recall score 
ax[1][0].plot(results.history['val_recall'],'b')
ax[1][0].plot(results.history['recall'],'r')
ax[1][0].set_ylabel("Recall Score")
ax[1][0].legend(['Validation set','Training set'])

# Plot Focal Loss 
ax[1][1].plot(results.history['val_loss'],'b')
ax[1][1].plot(results.history['loss'],'r')
ax[1][1].set_ylabel("Loss per epoch")
ax[1][1].set_xlabel("Epochs transcurred")
ax[1][1].legend(['Validation set','Training set'])

# Save plot for use in report 
plt.savefig("mtb_plot.png", dpi=300, bbox_inches="tight")
