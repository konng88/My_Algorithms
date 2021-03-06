import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.


    idx1 = generator.randint(n)
    idxs = [idx1]


    N,D = x.shape
    for k in range(1,n_cluster):
        min_dis = []
        for i in range(N):
            dis = np.linalg.norm(x[i]-x[idxs],ord=2,axis=1)
            min_dis.append(np.min(dis))
        idxs.append(np.argmax(min_dis))

    centers = idxs

    #
    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')



    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE


        centroids = x[self.centers]

        J = float('inf')

        # for iter in range(self.max_iter):
        #     r = np.zeros((N,self.n_cluster))
        #     print(iter)
        #     J_new = 0
        #     y = np.zeros((N,))
        #     for n in range(N):
        #         dis = np.linalg.norm(x[n]-centroids,ord=2,axis=1)
        #         k = np.argmin(dis)
        #         r[n][k] = 1
        #         y[n] = k
        #         J_new += dis[k]**2

        # if abs(J - J_new) <= self.e:
        #     break
        # J = J_new
        #
        # for k in range(self.n_cluster):
        #     centroids = (np.dot(r.T,x).T / np.sum(r,axis=0)).T


        for iter in range(self.max_iter):
            print(iter)
            x2 = np.dot((np.linalg.norm(x,ord=2,axis=1)**2).reshape((N,1)),np.ones((1,self.n_cluster)))

            c2 = np.dot(np.ones((N,1)),(np.linalg.norm(centroids,ord=2,axis=1)**2).reshape((1,self.n_cluster)))

            xc = np.dot(x,centroids.T)

            M = x2 + c2 - 2*xc

            J_new = np.sum(np.min(M,axis=1))
            y = np.argmin(M,axis=1)

            if abs(J - J_new) <= self.e:
                break
            J = J_new

            for i in range(self.n_cluster):
                centroids[i] = np.mean(x[y==i],axis=0)



        # centroids = x[self.centers]
        # membership = np.zeros(N).astype(int)
        # J = float('inf')
        #
        # for iter in range(self.max_iter):
        #     print(iter)
        #     J_new = 0
        #
        #     for i in range(N):
        #         dis = np.linalg.norm(x[i]-centroids,ord=2,axis=1)
        #         k = np.argmin(dis)
        #         membership[i] = k
        #         J_new += dis[k]**2
        #
        #     if abs(J - J_new) <= self.e:
        #         break
        #     J = J_new
        #
        #     for i in range(self.n_cluster):
        #         centroids[i] = np.mean(x[membership==i],axis=0)
        #
        # y = membership
        #





        # raise Exception(
        #      'Implement fit function in KMeans class')

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter




class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        # raise Exception(
        #      'Implement fit function in KMeansClassifier class')
        clf = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e, generator=self.generator)
        centroids,labels,iter = clf.fit(x=x,centroid_func=centroid_func)
        self.centroid_labels = np.zeros(self.n_cluster)
        self.centroids = centroids

        for i in range(0,self.n_cluster):
            cluster_labels = y[labels==i]
            info = np.unique(cluster_labels,return_counts=True)
            majority = info[0][np.argmax(info[1])]
            self.centroid_labels[i] = majority

        # DONOT CHANGE CODE BELOW THIS LINE





        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        labels = np.zeros(N)
        for i in range(N):
            Xi = np.dot(np.ones((self.n_cluster,1)),x[i].reshape((1,D)))
            norm = np.linalg.norm(Xi-self.centroids,ord=2,axis=1)
            labels[i] = self.centroid_labels[np.argmin(norm)]

        # raise Exception(
        #      'Implement predict function in KMeansClassifier class')


        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE

    new_im = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[0])):
            dis = np.linalg.norm(image[i][j] - code_vectors,ord=2,axis=1)
            k = np.argmin(dis)
            new_im[i][j] = code_vectors[k]


    # raise Exception(
    #          'Implement transform_image function')


    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
