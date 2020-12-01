from parameters import *
import numpy as np
import timeit


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.color is True:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
    else:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        # calculam media pe canale de culoare pentru fiecare poza mica
        small_images_mean = np.mean(params.small_images, axis=(1, 2))
        # matrice pe care o folosim in cazul vecinilor diferiti
        neighbors = np.full([params.num_pieces_vertical, params.num_pieces_horizontal], -1)

        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                # luam patch-ul curent in functie de i si j si ii calculam media
                current_image = np.mean(params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W], axis=(0, 1))
                # aplicam distanta euclidiana, in cazul grayscale nu avem o suma de calculat
                if params.color is True:
                    distances = np.sum(((small_images_mean - current_image) ** 2), axis=1)
                else:
                    distances = (small_images_mean - current_image) ** 2

                if params.neighbors is True:
                    # coordonatele relative ale vecinilor fata de patch-ul curent
                    new_neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                    for k in range(len(new_neighbors)):
                        new_i = i + new_neighbors[k][0]
                        new_j = j + new_neighbors[k][1]
                        # in cazul in care vecinul este in imagine si n-are valoarea -1
                        # excludem valoarea sa din lista solutiilor
                        if new_i in range(params.num_pieces_vertical) and new_j in range(params.num_pieces_horizontal):
                            if neighbors[new_i][new_j] != -1:
                                distances[neighbors[new_i][new_j]] = np.inf

                #alegem imaginea cu cea mai mica distanta si o punem in locul patch-ului
                min_index = np.argmin(distances)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[min_index]
                #completam matricea vecinilor cu indexul imaginii alese
                neighbors[i][j] = min_index
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.color is True:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
    else:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'distantaCuloareMedie':
        # matricea pixelilor ramasi neacoperiti
        done_matrix = np.array(range(h * w)).reshape((h, w))

        while np.any(done_matrix != -1):
            # alegem la intamplare al n-lea pixel, cu conditia sa fie neacoperit
            free_ = done_matrix[done_matrix > -1]
            index = np.random.randint(low=0, high=len(free_), size=1)
            # il transformam in index 2D pentru matricea h*w
            i = free_[index][0] // w
            j = free_[index][0] % w
            # (i,j) se refera la coltul din stanga sus si calculam unde cade coltul dreapta jos(i_max, j_max)
            # astfel incat sa nu iesim in afara pozei
            i_max = min(h, i + H)
            j_max = min(w, j + W)

            # decupam un patch din imaginea initiala si din imaginile mici conform dimensiunilor noi, dupa care
            # facem media culorilor
            current_image = np.mean(params.image_resized[i: i_max, j: j_max], axis=(0, 1))
            current_small_images = params.small_images[:, 0: i_max - i, 0: j_max - j]
            small_images_mean = np.mean(current_small_images, axis=(1, 2))

            # calculam distantele euclidiene
            if params.color is True:
                distances = np.sum(((small_images_mean - current_image) ** 2), axis=1)
            else:
                distances = (small_images_mean - current_image) ** 2
            # alegem imaginea cu cea mai mica distanta si o punem in locul patch-ului
            min_index = distances.argmin()
            img_mosaic[i: i_max, j: j_max] = current_small_images[min_index]
            # marcam toti pixelii din patch drept ocupati
            done_matrix[i: i_max, j: j_max] = -1
            print('Building mosaic %.5f%%' % (100 * (np.count_nonzero(done_matrix == -1)) / (h * w)))

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.color is True:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
    else:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    num_pieces = (params.num_pieces_vertical * 2 + 2) * (params.num_pieces_horizontal * 2 + 2)

    # pentru a crea masca prima data completam o matrice H*W cu valoarea 1
    if params.color is True:
        mask = np.full((H, W, C), 1)
    else:
        mask = np.full((H, W), 1)
    for i in range(H // 2):
        for j in range(W // 3 - i, -1, -1):
            # completam patru triunghiuri in colturi cu 0 pentru a obtine un hexagon in centru
            mask[i][j] = 0 # coltul stanga sus
            mask[i][W - j - 1] = 0 # coltul dreapta sus
            mask[H - i - 1][j] = 0 # coltul stanga jos
            mask[H - i - 1][W - j - 1] = 0 # coltul dreapta jos

    # cream o imagine mai mare din imaginea initiala, la final vom decupa din ea mozaicul
    if params.color is True:
        bigger_image = np.zeros((h + 2 * H, w + 2 * W, c))
    else:
        bigger_image = np.zeros((h + 2 * H, w + 2 * W))
    bigger_image[H:H + h, W:W + w] = params.image_resized

    if params.criterion == 'distantaCuloareMedie':
        # calculam media pe canale de culoare pentru fiecare poza mica
        small_images_mean = np.mean(params.small_images, axis=(1, 2))
        # matrice pe care o folosim in cazul vecinilor diferiti
        neighbors = np.full([params.num_pieces_vertical * 2 + 2, params.num_pieces_horizontal * 2 + 2], -1)

        # incepem cu liniile impare, acestea fiind defazate cu H/2 pe fata de liniile pare precendente
        first_row_start = H // 2
        row_index = 1
        for i in range(first_row_start, bigger_image.shape[0] - H, H):
            col_index = 0
            # distanta pe linie spre urmatoarea coloana este W + W/3(portiunea din lungime,
            # care nu este acoperita de masca)
            for j in range(0, bigger_image.shape[1] - W, W + W // 3):
                # distanta euclidiana este calculata pentru forma patratica a patchului
                current_image = np.mean(bigger_image[i:i + H, j:j + W], axis=(0, 1))
                if params.color is True:
                    distances = np.sum(((small_images_mean - current_image) ** 2), axis=1)
                else:
                    distances = (small_images_mean - current_image) ** 2

                if params.neighbors is True:
                    # un hexagon are 6 vecini posibili:
                    # sus, dreapta-sus, dreapta-jos, jos, stanga-jos, stanga-sus
                    new_neighbors = [[-2, 0], [-1, 1], [1, 1], [2, 0], [1, -1], [-1, -1]]
                    for k in range(len(new_neighbors)):
                        new_i = row_index + new_neighbors[k][0]
                        new_j = col_index + new_neighbors[k][1]
                        # in cazul in care vecinul este in imagine si n-are valoarea -1
                        # excludem valoarea sa din lista solutiilor
                        if new_i in range(params.num_pieces_vertical * 2 + 2) and new_j in range(params.num_pieces_horizontal * 2 + 2):
                            if neighbors[new_i][new_j] != -1:
                                distances[neighbors[new_i][new_j]] = np.inf

                # nu inlocuim tot patchul patrat cu imaginea gasita ci doar ce este in forma de hexagon
                # colturile sunt pastrate intacte cum sunt in bigger_image
                min_index = np.argmin(distances)
                bigger_image[i:i + H, j:j + W] = (1 - mask) * bigger_image[i:i + H, j:j + W] + mask * params.small_images[min_index]
                neighbors[row_index][col_index] = min_index
                # deplasarea pe linie inseamna sa sarim doua pozitii pe coloana
                col_index += 2
            # elementul care atinge marginea de jos este cu doua linii mai jos
            row_index += 2
            print('Building mosaic %.2f%%' % (50 * (i - H // 2) / (bigger_image.shape[0] - 2 * H)))

        # la fel procedam pentru liniile pare
        row_index = 0
        for i in range(0, bigger_image.shape[0] - H, H):
            col_index = 1
            # primul element este cu 2/3 mai la dreapta fata de 0( se intersecteaza pe un triunghi negru din masca)
            for j in range((W * 2) // 3, bigger_image.shape[1] - W, W + W // 3):
                current_image = np.mean(bigger_image[i:i + H, j:j + W], axis=(0, 1))
                if params.color is True:
                    distances = np.sum(((small_images_mean - current_image) ** 2), axis=1)
                else:
                    distances = (small_images_mean - current_image) ** 2

                if params.neighbors is True:
                    new_neighbors = [[-2, 0], [-1, 1], [1, 1], [2, 0], [1, -1], [-1, -1]]
                    for k in range(len(new_neighbors)):
                        new_i = row_index + new_neighbors[k][0]
                        new_j = col_index + new_neighbors[k][1]
                        if new_i in range((params.num_pieces_vertical + 1) * 2) and new_j in range((params.num_pieces_horizontal + 1) * 2):
                            if neighbors[new_i][new_j] != -1:
                                distances[neighbors[new_i][new_j]] = np.inf

                min_index = np.argmin(distances)
                bigger_image[i:i + H, j:j + W] = (1 - mask) * bigger_image[i:i + H, j:j + W] + mask * params.small_images[min_index]
                neighbors[row_index][col_index] = min_index
                col_index += 2
            row_index += 2
            print('Building mosaic %.2f%%' % (50 + (50 * i / (bigger_image.shape[0] - 2 * H))))
        img_mosaic = bigger_image[H:H + h, W:W + w]

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)
    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic
