import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.image as mpimg

def heatmap(descriptors):
    plt.figure(figsize = (20,20))
    sns.heatmap((descriptors).select_dtypes(include=['number']).corr(), annot = True, vmin=-1, vmax=1, center= 0)
    plt.title("heatmap признаков", fontsize=20)
    plt.show()

def display_images(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = mpimg.imread(img_path)
        plt.figure(figsize=(8, 8)) 
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename)
        plt.show()


def box_plot(data):
    plt.boxplot(data)
    plt.title("Ящик с усами")
    plt.xlabel("Данные")
    plt.ylabel("Значения")
    plt.show()

def bar_target(data):
    counts = data.value_counts()
    counts = counts.sort_index()
    plt.figure(figsize=(6, 4))
    color_list = ['skyblue', 'orange', 'red'][:len(counts)]
    plt.bar(counts.index.astype(str), counts.values, color=color_list, alpha=0.5)
    plt.xlabel('Значения')
    plt.ylabel('Количество')
    plt.title('Сбалансированнность классов')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()