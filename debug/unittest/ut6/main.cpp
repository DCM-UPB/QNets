#include "StringCodeUtilities.hpp"

#include <iostream>
#include <vector>
#include <assert.h>


int main(){
    using namespace std;

    static const int ncodes = 4;

    // These codes will be used as test input for string code methods

    static const string testTreeCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C { M ( b 1 ) { MM } , N , M ( b 0 ) }", "D ( s name ) { M }"};
    static const string testTreeCode_empty_brackets1[ncodes] = {"A ( )" , "B ( f 0.1 , i 2 )" , "C ( ) { M ( b 1 ) { MM ( ) } , N ( ) , M ( b 0 ) }", "D ( s name ) { M ( ) }"};
    static const string testTreeCode_empty_brackets2[ncodes] = {"A { }" , "B ( f 0.1 , i 2 ) { }" , "C { M ( b 1 ) { MM { } } , N { } , M ( b 0 ) { } }", "D ( s name ) { M { } }"};
    static const string testTreeCode_empty_brackets3[ncodes] = {"A ( ) { }" , "B ( f 0.1 , i 2 ) { }" , "C ( ) { M ( b 1 ) { MM ( ) { } } , N ( ) { } , M ( b 0 ) { } }", "D ( s name ) { M ( ) { } }"};

    // these codes will be used as comparison (and in the end as input as well)

    static const string testIdCode[ncodes] = {"A", "B", "C", "D"};
    static const string testParams[ncodes] = {"", "f 0.1 , i 2", "", "s name"};
    static const string testParamValue_i[ncodes] = {"", "2", "", ""};
    static const string testFullCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C", "D ( s name )"};

    static const string testMemberTreeCode[ncodes] = {"", "", "M ( b 1 ) { MM } , N , M ( b 0 )", "M"};
    static const string testTreeCode_M[ncodes] = {"", "", "M ( b 1 ) { MM }", "M"};
    static const string testTreeCode_M1[ncodes] = {"", "", "M ( b 0 )", ""};

    static const string testMemberTreeCode_empty_brackets1[ncodes] = {"", "", "M ( b 1 ) { MM ( ) } , N ( ) , M ( b 0 )", "M ( )"};
    static const string testTreeCode_M_empty_brackets1[ncodes] = {"", "", "M ( b 1 ) { MM ( ) }", "M ( )"};
    static const string testTreeCode_M1_empty_brackets1[ncodes] = {"", "", "M ( b 0 )", ""};

    static const string testMemberTreeCode_empty_brackets2[ncodes] = {"", "", "M ( b 1 ) { MM { } } , N { } , M ( b 0 ) { }", "M { }"};
    static const string testTreeCode_M_empty_brackets2[ncodes] = {"", "", "M ( b 1 ) { MM { } }", "M { }"};
    static const string testTreeCode_M1_empty_brackets2[ncodes] = {"", "", "M ( b 0 ) { }", ""};

    static const string testMemberTreeCode_empty_brackets3[ncodes] = {"", "", "M ( b 1 ) { MM ( ) { } } , N ( ) { } , M ( b 0 ) { }", "M ( ) { }"};
    static const string testTreeCode_M_empty_brackets3[ncodes] = {"", "", "M ( b 1 ) { MM ( ) { } }", "M ( ) { }"};
    static const string testTreeCode_M1_empty_brackets3[ncodes] = {"", "", "M ( b 0 ) { }", ""};

    static const string testDropParams[ncodes] = {"A", "B", "C { M { MM } , N , M }", "D { M }"};
    static const string testDropParams_empty_brackets[ncodes] = {"A { }", "B { }", "C { M { MM { } } , N { } , M { } }", "D { M { } }"};

    static const string testDropMembers[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C", "D ( s name )"};
    static const string testDropMembers_empty_brackets[ncodes] = {"A ( )" , "B ( f 0.1 , i 2 )" , "C ( )", "D ( s name )"};

    static const string testDropMembers_lvl2[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C { M ( b 1 ) , N , M ( b 0 ) }", "D ( s name ) { M }"};
    static const string testDropMembers_lvl2_empty_brackets1[ncodes] = {"A ( )" , "B ( f 0.1 , i 2 )" , "C ( ) { M ( b 1 ) , N ( ) , M ( b 0 ) }", "D ( s name ) { M ( ) }"};
    static const string testDropMembers_lvl2_empty_brackets2[ncodes] = {"A { }" , "B ( f 0.1 , i 2 ) { }" , "C { M ( b 1 ) , N , M ( b 0 ) }", "D ( s name ) { M }"};
    static const string testDropMembers_lvl2_empty_brackets3[ncodes] = {"A ( ) { }" , "B ( f 0.1 , i 2 ) { }" , "C ( ) { M ( b 1 ) , N ( ) , M ( b 0 ) }", "D ( s name ) { M ( ) }"};

    static const int testCountNParams[ncodes] = {0, 2, 0, 1};
    static const int testCountTreeNParams[ncodes] = {0, 2, 2, 1};
    static const int testCountDirectNMembers[ncodes] = {0, 0, 3, 1};
    static const int testCountTreeNMembers[ncodes] = {0, 0, 4, 1};


    // create input treecode vectors
    vector<const string *> testTreeCode_vec;
    testTreeCode_vec.push_back(&testTreeCode[0]);
    testTreeCode_vec.push_back(&testTreeCode_empty_brackets1[0]);
    testTreeCode_vec.push_back(&testTreeCode_empty_brackets2[0]);
    testTreeCode_vec.push_back(&testTreeCode_empty_brackets3[0]);


    // create comparison vectors
    vector<const string *> testMemberTreeCode_vec;
    testMemberTreeCode_vec.push_back(&testMemberTreeCode[0]);
    testMemberTreeCode_vec.push_back(&testMemberTreeCode_empty_brackets1[0]);
    testMemberTreeCode_vec.push_back(&testMemberTreeCode_empty_brackets2[0]);
    testMemberTreeCode_vec.push_back(&testMemberTreeCode_empty_brackets3[0]);

    vector<const string *> testTreeCode_M_vec;
    testTreeCode_M_vec.push_back(&testTreeCode_M[0]);
    testTreeCode_M_vec.push_back(&testTreeCode_M_empty_brackets1[0]);
    testTreeCode_M_vec.push_back(&testTreeCode_M_empty_brackets2[0]);
    testTreeCode_M_vec.push_back(&testTreeCode_M_empty_brackets3[0]);

    vector<const string *> testTreeCode_M1_vec;
    testTreeCode_M1_vec.push_back(&testTreeCode_M1[0]);
    testTreeCode_M1_vec.push_back(&testTreeCode_M1_empty_brackets1[0]);
    testTreeCode_M1_vec.push_back(&testTreeCode_M1_empty_brackets2[0]);
    testTreeCode_M1_vec.push_back(&testTreeCode_M1_empty_brackets3[0]);

    vector<const string *> testDropParams_vec;
    testDropParams_vec.push_back(&testDropParams[0]);
    testDropParams_vec.push_back(&testDropParams[0]);
    testDropParams_vec.push_back(&testDropParams_empty_brackets[0]);
    testDropParams_vec.push_back(&testDropParams_empty_brackets[0]);

    vector<const string *> testDropMembers_vec;
    testDropMembers_vec.push_back(&testDropMembers[0]);
    testDropMembers_vec.push_back(&testDropMembers_empty_brackets[0]);
    testDropMembers_vec.push_back(&testDropMembers[0]);
    testDropMembers_vec.push_back(&testDropMembers_empty_brackets[0]);

    vector<const string *> testDropMembers_lvl2_vec;
    testDropMembers_lvl2_vec.push_back(&testDropMembers_lvl2[0]);
    testDropMembers_lvl2_vec.push_back(&testDropMembers_lvl2_empty_brackets1[0]);
    testDropMembers_lvl2_vec.push_back(&testDropMembers_lvl2_empty_brackets2[0]);
    testDropMembers_lvl2_vec.push_back(&testDropMembers_lvl2_empty_brackets3[0]);


    // empty, single, multi element test vectors
    static const vector<string> empty_vec;
    static const vector<string> single_vec = {"A"};
    static const vector<string> multi_vec = {"A", "B", "C"};

    // params code with params of all usual types and the corresponding parameters with different initialization
    static const string fparam = "f 0.1";
    static const string iparam = "i 2";
    static const string bparam = "b 1";
    static const string sparam = "s name";
    static const string allParams = fparam + " , " + iparam + " , " + bparam + " , " + sparam;

    double f = 0., f_test = 0.1;
    int i = 0, i_test = 2;
    bool b = false, b_test = true;
    string s = "", s_test = "name";


    // working variables
    string str; // for storing strings
    int counter; // for storing counters

    for (int j=0; j<ncodes; ++j) {
        // now go through different levels of useless brackets use
        int it = 0;
        for ( const string * testArray : testTreeCode_vec ) {

            // --- READERS

            // readIdCode
            //cout << readIdCode(testArray[j]) << endl;
            //cout << testIdCode[j] << endl << endl;
            assert(readIdCode(testArray[j]) == testIdCode[j]);

            // readParams / readParamValue
            str = readParams(testArray[j]);
            //cout << str << endl;
            //cout << testParams[j] << endl << endl;
            assert(str == testParams[j]);

            //cout << readParamValue(str, "i") << endl;
            //cout << testParamValue_i[j] << endl << endl;
            assert(readParamValue(str, "i") == testParamValue_i[j]);

            // read(Member)TreeCode
            str = readMemberTreeCode(testArray[j]);
            //cout << str  << endl;
            //cout << testMemberTreeCode_vec[it][j]  << endl << endl;

            assert(str == testMemberTreeCode_vec[it][j]);

            //cout << readTreeCode(str, "M")  << endl;
            //cout << testTreeCode_M_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, "M") == testTreeCode_M_vec[it][j]);

            //cout << readTreeCode(str, "M", 1)  << endl;
            //cout << testTreeCode_M1_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, "M", 1) == testTreeCode_M1_vec[it][j]);


            // --- DROP

            // dropParams
            //cout << dropParams(testArray[j])  << endl;
            //cout << testDropParams_vec[it][j]  << endl << endl;
            assert(dropParams(testArray[j]) == testDropParams_vec[it][j]);

            // dropMembers
            //cout << dropMembers(testArray[j])  << endl;
            //cout << testDropMembers_vec[it][j]  << endl << endl;
            assert(dropMembers(testArray[j]) == testDropMembers_vec[it][j]);

            //cout << dropMembers(testArray[j], 2)  << endl;
            //cout << testDropMembers_lvl2_vec[it][j]  << endl << endl;
            assert(dropMembers(testArray[j], 2) == testDropMembers_lvl2_vec[it][j]);


            // --- COUNTERS

            // countNParams
            counter = countNParams(readParams(testArray[j]));
            //cout << counter << " <-> " << testCountNParams[j] << endl << endl;
            assert(counter == testCountNParams[j]);

            // countTreeNParams
            counter = countTreeNParams(testArray[j]);
            //cout << counter << " <-> " << testCountTreeNParams[j] << endl << endl;
            assert(counter == testCountTreeNParams[j]);

            // countNMembers (direct)
            str = readMemberTreeCode(testArray[j]);
            counter = countNMembers(str); // default is direct_only=true
            //cout << counter << " <-> " << testCountDirectNMembers[j] << endl << endl;
            assert(counter == testCountDirectNMembers[j]);

            // countNMembers (tree)
            counter = countNMembers(str, false); // count through tree
            //cout << counter << " <-> " << testCountTreeNMembers[j] << endl << endl;
            assert(counter == testCountTreeNMembers[j]);



            ++it;
        }
        //cout << "----------------------------------------------------" << endl << endl;
    }

    // --- COMPOSERS
    //cout << "Composers:" << endl << endl;


    // composeCodes

    //cout << composeCodes("", "") << endl;
    //cout << ""  << endl << endl;
    assert(composeCodes("", "") == "");

    //cout << composeCodes(testIdCode[0], "") << endl;
    //cout << testIdCode[0] << endl << endl;
    assert(composeCodes(testIdCode[0], "") == testIdCode[0]);

    //cout << composeCodes("", testIdCode[1]) << endl;
    //cout << testIdCode[1] << endl << endl;
    assert(composeCodes("", testIdCode[1]) == testIdCode[1]);

    //cout << composeCodes(testIdCode[0], testIdCode[1]) << endl;
    //cout << testIdCode[0] + " , " + testIdCode[1] << endl << endl;
    assert(composeCodes(testIdCode[0], testIdCode[1]) == testIdCode[0] + " , " + testIdCode[1]);


    // composeCodeList

    //cout << composeCodeList(empty_vec) << endl;
    //cout << "" << endl << endl;
    assert(composeCodeList(empty_vec) == "");

    //cout << composeCodeList(single_vec) << endl;
    //cout << "A" << endl << endl;
    assert(composeCodeList(single_vec) == "A");

    //cout << composeCodeList(multi_vec) << endl;
    //cout << "" << endl << endl;
    assert(composeCodeList(multi_vec) == "A , B , C");


    // composeFullCode

    //cout << composeFullCode(testIdCode[0], testParams[0]) << endl;
    //cout << testFullCode[0] << endl << endl;
    assert(composeFullCode(testIdCode[0], testParams[0]) == testFullCode[0]); // empty params

    //cout << composeFullCode(testIdCode[1], testParams[1]) << endl;
    //cout << testFullCode[1] << endl << endl;
    assert(composeFullCode(testIdCode[1], testParams[1]) == testFullCode[1]); // with params


    // composeTreeCode

    //cout << composeTreeCode(testFullCode[0], testMemberTreeCode[0]) << endl;
    //cout << testTreeCode[0] << endl << endl;
    assert(composeTreeCode(testFullCode[0], testMemberTreeCode[0]) == testTreeCode[0]); // empty members

    //cout << composeTreeCode(testFullCode[3], testMemberTreeCode[3]) << endl;
    //cout << testTreeCode[3] << endl << endl;
    assert(composeTreeCode(testFullCode[3], testMemberTreeCode[3]) == testTreeCode[3]); // with member


    // TEMPLATES
    //cout << "Templates:" << endl << endl;

    // setParamValue

    assert(setParamValue(allParams, "f", f));
    //cout << f << " <-> " << f_test << endl;
    assert(f == f_test);

    assert(setParamValue(allParams, "i", i));
    //cout << f << " <-> " << i_test << endl;
    assert(f == f_test);

    assert(setParamValue(allParams, "b", b));
    //cout << f << " <-> " << b_test << endl;
    assert(f == f_test);

    assert(setParamValue(allParams, "s", s));
    //cout << f << " <-> " << s_test << endl;
    assert(f == f_test);

    //cout << endl;

    // composeParamCode

    //cout << composeParamCode("f", f_test) << endl;
    //cout << fparam << endl << endl;
    assert(composeParamCode("f", f_test) == fparam);

    //cout << composeParamCode("i", i_test) << endl;
    //cout << iparam << endl << endl;
    assert(composeParamCode("i", i_test) == iparam);

    //cout << composeParamCode("b", b_test) << endl;
    //cout << bparam << endl << endl;
    assert(composeParamCode("b", b_test) == bparam);

    //cout << composeParamCode("s", s_test) << endl;
    //cout << sparam << endl << endl;
    assert(composeParamCode("s", s_test) == sparam);

    //cout << endl;

    return 0;
}
