# new-tracker
A new arbitrary object tracker without recurrence

For a test run, extract the bzip2 archive [here](https://drive.google.com/open?id=0B04f_ATQk8h1OXNvVzMzbUxEUXc)
into the current directory and run
```
python main.py --ilsvrc ILSVRC
```

If you don't have GPU, set the environment variable `NOCUDA` to 1:
```
NOCUDA=1 python main.py --ilsvrc ILSVRC
```

If you want to actually show the figures for visualization instead of saving them, run
```
APPDEBUG=1 python main.py --ilsvrc ILSVRC
```
