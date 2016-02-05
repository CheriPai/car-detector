# Path to store extracted features
pos_fd_path = 'data/pos_fd'
neg_fd_path = 'data/neg_fd'
pos_fd_test_path = 'data/pos_test'
neg_fd_test_path = 'data/neg_test'

# Path to store model
model_path = 'data/model/car_detector_model.pkl'

# Sliding window parameters
(win_width, win_height) = (100, 40)
step_size = 10


# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)
visualise = False
normalise = True

# Pyramid scale amount
scale_amt = 1.5
