import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import os,glob

color=[(223 / 255, 25 / 255, 27 / 255),(54 / 255, 123 / 255, 180 / 255), (75 / 255, 171 / 255, 72 / 255),
             (249 / 255, 124 / 255, 0 / 255)]
marker=['o', '^', 's','P']

def sigma_sensitivity(x,y,title='sigma_sensitivity',label=None,save=False, std=None,xlabel=None):
    # acc_cocitation_cora = np.array([0.67235,0.68442,0.69163,0.69217, 0.69011,0.69171,0.69147,0.68921,0.68598,0.68489, 0.68532])*100
    # x=np.array([-2,-1.5,-1,-0.5,-0.1,0,0.1, 0.5,1,1.5,2])

    ## sigma 灵敏度分析
    # title = 'cocitation-cora'
    ylabel = 'Acc'
    # y=acc_cocitation_cora
    if std is None:
        plt.plot(x,y,label=label, linewidth=1, linestyle='--',  marker=marker[1])
    else:
        plt.errorbar(x, y, label=label,yerr=std, fmt="o", linewidth=1, linestyle='--',elinewidth=1.7)
    plt.xlabel(xlabel, fontsize=19)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid()
    if label is not None:
        plt.legend(fontsize=16,)# loc='upper right')
    if save:
        # plt.savefig(dir_save + title + '.png')
        plt.savefig(f'./{title}.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f'./{title}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.0)
    # plt.show()
    # plt.close()

def spectral_anaysis():
    save = False
    title = 'cocitation-cora'
    xlabel = 'Index'
    ylabel = r'$\lambda$'
    for i,file in enumerate(glob.glob(os.path.join('../../model/hgnn/npz', 'eign_sigma*.npz'))):
        data=np.load(file)
        sigma,y = data['sigma'],data['e_vals']
        if sigma not in [-1,-0.5,0.0,1]:
            continue

        x = np.arange(0, len(y))
        plt.plot(x,y,label=fr'$\sigma$ = {sigma}', linewidth=1., linestyle='--')
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend()
    plt.grid()
    if save:
        # plt.savefig(dir_save + title + '.png')
        plt.savefig(f'./{title}.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f'./{title}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.show()

def trans(mean, sigma, std):
    '''
     对mean 和std 根据sigma 大小关系重排序
    '''
    return np.array(mean)[np.array(sigma).argsort()], np.array(std)[np.array(sigma).argsort()]

if __name__=='__main__':
    # v0-old
    # acc_cocitation_cora = np.array([0.67235,0.68442,0.69163,0.69217, 0.69011,0.69171,0.69147,0.68921,0.68598,0.68489, 0.68532])*100
    # acc_cocitation_citeseer = np.array([0.61979,0.62426,0.62423,0.62224, 0.61853,
    #                                     0.62001,0.61982,0.61749,0.61645,0.61575, 0.61667])*100
    # acc_cocitation_pubmed = np.array([0.74657,0.75126,0.75349,0.75087, 0.74975,
    #                                     0.74870,0.74795,0.74435,0.74174,0.73986, 0.73864])*100
    # acc_coauthorship_cora = np.array([0.69731,0.73785,0.74840,0.73333, 0.72765,
    #                                     0.72551,0.72523,0.72333,0.72290,0.72239, 0.72220])*100
    ## sigma sensitivity
    # sigma = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    # acc_cocitation_citeseer = np.array([0.64685, 0.6488 , 0.65148, 0.65135, 0.65032, 0.65066, 0.64953,
    #     0.64776, 0.64619])
    # acc_cocitation_citeseer_std = np.array([0.00846, 0.01331, 0.01025, 0.00972, 0.00969, 0.01069, 0.01104,
    #     0.01153, 0.01133])*0.51
    # acc_cocitation_cora = np.array([0.68263, 0.68921, 0.69221, 0.69256, 0.68999, 0.69085, 0.69147,
    #     0.68657, 0.6854 ])
    # acc_cocitation_cora_std =  np.array([0.01529, 0.01622, 0.01611, 0.01901, 0.01469, 0.02004, 0.01689,
    #     0.02381, 0.0183 ])*0.51
    # acc_coauthorship_cora = np.array([0.69595, 0.73742, 0.75755, 0.75954, 0.75911, 0.75752, 0.75576, 0.75463, 0.7535 ])
    # acc_coauthorship_cora_std = np.array([0.01292, 0.00981, 0.00953, 0.00835, 0.0075 , 0.00766, 0.00733, 0.0075 , 0.00748])*0.51
    # acc_cocitation_pubmed=np.array([0.74278, 0.74613, 0.74845, 0.7489 , 0.74905, 0.74812, 0.74757, 0.74679, 0.74596])
    # acc_cocitation_pubmed_std=np.array([0.01287, 0.01304, 0.01263, 0.01246, 0.01302, 0.01312, 0.01345, 0.01364, 0.01396])
    # xlabel = r'$\sigma$';title = r'sensitivityOf_sigma'
    #
    #
    # beta = np.array([0. , 0.2, 0.4, 0.6, 0.8,0.9, 1. ])
    # acc_cocitation_pubmed =  array([0.72637, 0.73117, 0.73638, 0.74172, 0.74596, 0.74773, 0.74905])
    # acc_cocitation_pubmed_std = array([0.01478, 0.01455, 0.01358, 0.01321, 0.01352, 0.01301, 0.01302])
    # acc_coauthorship_cora=array([0.58863, 0.64428, 0.69801, 0.73629, 0.75709, 0.75954, 0.75907])
    # acc_coauthorship_cora_std=np.array([0.01582, 0.01509, 0.01545, 0.01159, 0.00943, 0.00835, 0.00662])
    # acc_cocitation_cora=array([0.55514, 0.55432, 0.58053, 0.62185, 0.66776, 0.6926, 0.70643])
    # acc_cocitation_cora_std= array([0.02456, 0.02245, 0.02804, 0.02958, 0.0185, 0.01819, 0.01847])
    # xlabel = r'$\beta$';title = r'sensitivityOf_beta'
    # sigma=beta

    acc_cocitation_pubmed = array([0.75075, 0.74946, 0.74862, 0.74799, 0.74729, 0.74638, 0.7448 ])[::-1]
    acc_cocitation_pubmed_std =  array([0.01138, 0.01249, 0.01291, 0.01332, 0.01338, 0.01368, 0.01404])[::-1]
    acc_cocitation_cora = array([0.69907, 0.69591, 0.68458, 0.67114, 0.65584, 0.6454 , 0.63493])[::-1]
    acc_cocitation_cora_std = array([0.01971, 0.02056, 0.01827, 0.01707, 0.01771, 0.01551, 0.01438])[::-1]
    acc_coauthorship_cora =  array([0.76048, 0.75923, 0.75631, 0.75288, 0.74579, 0.73894, 0.7301 ])[::-1]
    acc_coauthorship_cora_std =  array([0.00746, 0.00876, 0.00913, 0.0112 , 0.0104 , 0.01093, 0.01206])[::-1]

    #ntu= array([0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.15,0.2, 0.25, 0.3]),
    # array([0.84611, 0.84531, 0.84611, 0.84718, 0.84718, 0.84906, 0.84799, 0.84987, 0.84933, 0.84987, 0.85121, 0.85121, 0.85013, 0.8504]),
    # array([0.00246, 0.00209, 0.00214, 0.0017, 0.00208, 0.00269, 0.00241, 0.0024, 0.00201, 0.00268, 0.00134, 0.00247, 0.00188, 0.00356])
    # modelNet = array([0.97699, 0.97674, 0.97642, 0.97674, 0.97694, 0.97735, 0.97759,
    #        0.97767, 0.97747, 0.97727, 0.97658, 0.97682, 0.97711, 0.97694]),
    # array([0.00047, 0.00055, 0.0007, 0.00049, 0.00053, 0.00053, 0.00057,
    #        0.00053, 0.00066, 0.00053, 0.00044, 0.00092, 0.00089, 0.00078])

    ntu = array([0.84611,  0.84906,0.84987, 0.85121, 0.85121, 0.85013, 0.8504])[::-1]
    ntu_std= array([0.00246, 0.00269,  0.00268, 0.00134, 0.00247, 0.00188, 0.00356])[::-1]
    alpha= (1-array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 ]))[::-1]
    modelNet = array([0.97699, 0.97735, 0.97727, 0.97658, 0.97682, 0.97711, 0.97694])[::-1]
    modelNet_std = array([0.00047, 0.00053,  0.00053, 0.00044, 0.00092, 0.00089, 0.00078])[::-1]
    xlabel = r'$\alpha$';title = r'sensitivityOf_alpha'

    sigma=alpha
    # sigma_sensitivity(sigma,y=acc_cocitation_cora,label='cora (cocitation)',std=acc_cocitation_cora_std,xlabel=xlabel)
    # sigma_sensitivity(sigma,y=acc_cocitation_citeseer,label='cocitation-citeseer',std=acc_cocitation_citeseer_std*0.51)
    # sigma_sensitivity(sigma,y=acc_cocitation_pubmed,label='pubmed (cocitation)',std=acc_cocitation_pubmed_std,xlabel=xlabel)
    sigma_sensitivity(sigma,y=ntu,label='NTU',std=ntu_std,xlabel=xlabel)
    sigma_sensitivity(sigma,y=modelNet,label='ModelNet40',std=modelNet_std,xlabel=xlabel)
    # sigma_sensitivity(sigma,y=acc_coauthorship_cora,label='cora (coauthorship)',std=acc_coauthorship_cora_std,xlabel=xlabel)
    save=True; # 调整坐标大小
    plt.ylim((0.8, 0.9815))
    # plt.xlim((-2, 3))
    plt.MultipleLocator(0.001)

    plt.grid(which = 'minor')

    if save:
        # plt.savefig(dir_save + title + '.png')
        plt.savefig(f'./{title}.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f'./{title}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.show()
