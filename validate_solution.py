import sys
import subprocess
import logging

titanic_str = '''
 _______ _ _              _      
|__   __(_) |            (_)     
   | |   _| |_ __ _ _ __  _  ___ 
   | |  | | __/ _` | '_ \| |/ __|
   | |  | | || (_| | | | | | (__ 
   |_|  |_|\__\__,_|_| |_|_|\___|

'''

if __name__ == '__main__':
    # Read in solution file
    try:
        solution_file = sys.argv[1]
    except IndexError:
        print(titanic_str)
        print('Error. Please pass the file path to your solution.')
        print('Usage: %s SOLUTION_FILE_PATH' % sys.argv[0])
        sys.exit(1)

    # Run solution code against dummy data
    dummy_train = 'titanic.csv'
    dummy_test = 'dummy-test.csv'

    # Get number of test samples. -1 for header row
    test_count = sum(1 for line in open(dummy_test)) - 1

    try:
        out = subprocess.run(['python3', solution_file, dummy_train, dummy_test], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)
    except Exception as e:
        print(titanic_str)
        print('------------------------------------------')
        print('There is an error with your code! Please fix before submitting.')
        print('------------------------------------------')
        logging.exception('Solution code error')
        print('------------------------------------------')
        print('There is an error with your code! Please fix before submitting.')
        print('------------------------------------------')

    # Check stdout contains dummy_test predictions
    lines = out.stdout.splitlines()
    solution_valid = True
    for line in lines:
        line = line.decode('utf-8')
        if line != '0' and line != '1':
            solution_valid = False

    # Check length of results
    if len(lines) != test_count:
        solution_valid = False

    print(titanic_str)
    if solution_valid:
        print('Solution valid and ready for submission.')
    else:
        print('Error. Please ensure your code outputs a list of 0s and 1s, one number per line for each test sample.')
        print('Current output from your code: ')
        print('------------------------------------------')
        print(out.stdout.decode('utf-8'))
        print('------------------------------------------')
        print('Please correct your code before submission.')

