import os
import cv2

class Visualizer():
    def __init__(self, *args):
        self.image_name = os.path.basename(os.path.splitext(args[0])[0])
        self.suffix = args[1]
        self.out_dir = args[2]
        self.meta_info = args[3]

    def visualize_data(self):
        file_name = self.image_name + '_' + self.suffix + '.jpg'
        out_file_dir = os.path.join(self.out_dir, file_name)
        cv2.imwrite(out_file_dir, self.meta_info)

    def visualize_dwt_data(self):
        image_channels = ['blue', 'green', 'red']

        titles = [
            'Approximation_LL', 
            'Vertical detail_LH',
            'Horizontal detail_HL',
            'Diagonal detail_HH'
        ]
        
        for channel in range (len(self.meta_info)):
            channel_name = image_channels[channel]
            channel_coefficients = self.meta_info[channel]
            LL, (LH, HL, HH) = channel_coefficients
            # [LL, LH, HL, HH] = channel_coefficients[0], channel_coefficients[1][0], channel_coefficients[1][1], channel_coefficients[1][2]

            for index, coefficients in enumerate([LL, LH, HL, HH]):
                import matplotlib.pyplot as plt

                # coefficients = coefficients.astype(np.uint8)
                # cc=image.astype(np.uint8)
                # im = cv2.cvtColor(cc, cv2.COLOR_BGR2RGB)

                print (coefficients.dtype)
                print (index)
                coefficient_name = titles[index]

                file_name = self.image_name + '_' + self.suffix + '_' + channel_name + '_' + coefficient_name + '.jpg'

                out_file_dir = os.path.join(self.out_dir, file_name)
                plt.savefig(coefficient_name + '.png')
                # cv2.imwrite(out_file_dir, coefficients)

        


        