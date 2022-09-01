static void badSink(int * data)
        delete [] data;
    int * data;
    /* Initialize data*/
    data = NULL;
    data = new int;
    badSink(data);
static void goodB2G1Sink(int * data)
        delete data;
    int * data;
    /* Initialize data*/
    data = NULL;
    data = new int;
    goodB2G1Sink(data);
static void goodB2G2Sink(int * data)
        delete data;
    int * data;
    /* Initialize data*/
    data = NULL;
    data = new int;
    goodB2G2Sink(data);
static void goodG2BSink(int * data)
        delete [] data;
    int * data;
    /* Initialize data*/
    data = NULL;
    data = new int[100];
    goodG2BSink(data);
    data = NULL; /* Initialize data */