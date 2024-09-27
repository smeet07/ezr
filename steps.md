To test out the difference between dumb(ASis) and smart(active learning) run:

    python3.13 -B extend.py data/optimize/config/*.csv

To run the tests:

    python3 tests.py

To run part1 of the experiment do:

    python3.13 -B extend3.py -t data/optimize/config/SS-T.csv //here filename can be replaced with any of the files
    ![image](https://github.com/user-attachments/assets/d5ade092-a025-4e00-89fa-7602d362b4a6)


We have used makefile to run the same command for all of the large files and store their outputs
To run part2 of the experiment there is a makefile create. to generate the outputs:
    make Act=branch outputs

Go to path to view the outputs:

    cd ~/tmp/branch/outputs
    ls
    ![image](https://github.com/user-attachments/assets/78a8d351-60a4-42b5-a6eb-934c783185b1)


for some reason the rq.sh did not work and showed "Column: line too long"
I have copy pasted the output files from the tmp directory to result directory. To view the summary, I have translated the rq.sh into summary.py. to get the summary:

    cd result //considering you are in home directory
    python3 summary.py
    ![image](https://github.com/user-attachments/assets/8265b53e-29c2-4953-a838-50e3627c255a)


I have captured the output in output.txt file
    cat output.txt

Optional :
    To see why rq.sh is not working. You can back into cd ~/tmp/branch/outputs run bash /ezr/process_csv.sh(relative path might be different in codespaces. It's basically in the root directory)
    ![image](https://github.com/user-attachments/assets/963a3f87-44f5-4686-8156-6b4c955657a4)


    



