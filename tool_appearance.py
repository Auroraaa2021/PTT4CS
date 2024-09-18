import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    for i in range(20, 26):
        file_path = '/home/data/CATARACTS/ground_truth/CATARACTS_2018/images/micro_train_gt/train' + str(i).zfill(2) + '.csv'
        video_num = "train" + str(i).zfill(2)
        
        df = pd.read_csv(file_path)

        frame_numbers = df.iloc[:, 0]
        tool_data = df.iloc[:, 2:]


        plt.figure(figsize=(18, 8))
        sns.heatmap(tool_data.T, cmap="YlGnBu", cbar=True, xticklabels=frame_numbers, yticklabels=tool_data.columns)
        plt.xticks(ticks=range(0, len(frame_numbers), 1000), labels=frame_numbers[::1000])
        plt.xlabel('Frame Number')
        plt.ylabel('Tools')
        plt.title('Tool Presence Across Frames')
        
        # Save the figure
        output_path = './output/tool_presence_' + video_num + '.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.show()