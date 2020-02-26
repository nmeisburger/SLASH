#include "CMS.h"
#include "LSH.h"
#include "LSHReservoirSampler.h"
#include "benchmarking.h"
#include "dataset.h"
#include "flashControl.h"
#include "indexing.h"
#include "mathUtils.h"
#include "omp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {

#ifdef TEST_RUN
    testing()
#endif
#ifdef FILE_OUTPUT
        evalWithFileOutput();
#endif
#ifdef EVAL_SIM
    evalWithSimilarity();
#endif
#ifdef WIKIDUMP
    wikiDump();
#endif
#ifdef CRITEO
    criteo();
#endif

    return 0;
}
