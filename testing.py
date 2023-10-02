'''YES'''

import ast
import time
from numpy import ndarray
from feature_extraction import number_of_loops
from helpers import extract_images, get_black_white


# pylint: disable=invalid-name
# pylint: disable=unused-variable


def test_num_loops() -> None:
    '''Tests the `number_of_loops()` function.'''

    def my_num_loops(image: ndarray) -> int:
        '''
        `Feature 7` returns the number of loops in the given `image`.
        '''

        BLACK, WHITE = 0, 1

        visited = {}
        for row in range(28):
            for col in range(28):
                if int(image[row][col]) == BLACK:
                    # False indicates we have not seen the indices yet.
                    visited[str([row, col])] = False
                elif int(image[row][col]) == WHITE:
                    # True indicates we have seen the indices.
                    visited[str([row, col])] = True
                else:
                    raise ValueError(f'Invalid pixel value: {image[row][col]}')

        def is_valid(x, y):
            return 0 <= x < 28 and 0 <= y < 28

        def bfs(key):
            bfs_queue = [ast.literal_eval(key)]

            directions = [
                [1, -1], [1, 0], [1, 1],
                [0, -1],          [0, 1],
                [-1, -1], [-1, 0], [-1, 1]]

            while len(bfs_queue) > 0:
                x, y = bfs_queue.pop()
                visited[str([x, y])] = True

                for (d_x, d_y) in directions:
                    n_x, n_y = x + d_x, y + d_y

                    if is_valid(n_x, n_y) and int(image[n_x][n_y]) == BLACK and not visited[str([n_x, n_y])]:
                        bfs_queue.append((n_x, n_y))
                        visited[str([n_x, n_y])] = True

        search_count = 0
        for key, _ in visited.items():
            if not visited[key]:
                bfs(key)
                search_count += 1

        # Subtract 1 since we counted the background as a loop.
        return search_count - 1

    my_longest_time, jacks_longest_time = 0, 0

    for digit in range(NUM_FILES := 10):

        IMAGES, _ = extract_images(
            file=f'input_files/training_data/handwritten_samples_{digit}.csv',
            has_label=True)

        BINARY_IMAGES = [get_black_white(image) for image in IMAGES]

        for i, image in enumerate(BINARY_IMAGES):
            start = time.perf_counter()
            mine = my_num_loops(image)
            end = time.perf_counter()
            my_time = end - start

            start = time.perf_counter()
            jacks = number_of_loops(image)
            end = time.perf_counter()
            jacks_time = end - start

            my_longest_time = max(my_longest_time, my_time)
            jacks_longest_time = max(jacks_longest_time, jacks_time)

            print(mine)

            assert mine == jacks, f'image {i}: {mine}, {jacks}'

    # print('All tests passed !')
    # print('-' * 70)
    # print(f'My longest time:\t{my_longest_time * 1000:.6f} ms')
    # print(f'Jack\'s longest time:\t{jacks_longest_time * 1000:.6f} ms')


def main() -> None:
    '''Le Main'''

    test_num_loops()


if __name__ == '__main__':
    main()
