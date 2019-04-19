#include "qnets/poly/serial/StringCodeUtilities.hpp"

#include <cassert>


int main()
{
    using namespace std;

    const int ncodes = 4;

    // These codes will be used as test input for string code methods

    const string testTreeCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C { M ( b 1 ) { M , N } , N , M ( b 0 ) }", "D ( s N , i 1 ) { M }"};
    const string testTreeCode_empty_brackets1[ncodes] = {"A ( )", "B ( f 0.1 , i 2 )", "C ( ) { M ( b 1 ) { M ( ) , N ( ) } , N ( ) , M ( b 0 ) }", "D ( s N , i 1 ) { M ( ) }"};
    const string testTreeCode_empty_brackets2[ncodes] = {"A { }", "B ( f 0.1 , i 2 ) { }", "C { M ( b 1 ) { M { } , N { } } , N { } , M ( b 0 ) { } }", "D ( s N , i 1 ) { M { } }"};
    const string testTreeCode_empty_brackets3[ncodes] = {"A ( ) { }", "B ( f 0.1 , i 2 ) { }", "C ( ) { M ( b 1 ) { M ( ) { } , N ( ) { } } , N ( ) { } , M ( b 0 ) { } }", "D ( s N , i 1 ) { M ( ) { } }"};

    // these codes will be used as comparison (and in the end as input as well)

    const string testIdCode[ncodes] = {"A", "B", "C", "D"};
    const string testParams[ncodes] = {"", "f 0.1 , i 2", "", "s N , i 1"};
    const string testParamValue_i[ncodes] = {"", "2", "", "1"};
    const string testFullCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C", "D ( s N , i 1 )"};

    const string testMemberTreeCode[ncodes] = {"", "", "M ( b 1 ) { M , N } , N , M ( b 0 )", "M"};
    const string testTreeCode_M[ncodes] = {"", "", "M ( b 1 ) { M , N }", "M"};
    const string testTreeCode_M1[ncodes] = {"", "", "M ( b 0 )", ""};
    const string testTreeCode_1[ncodes] = {"", "", "N", ""};

    const string testMemberTreeCode_empty_brackets1[ncodes] = {"", "", "M ( b 1 ) { M ( ) , N ( ) } , N ( ) , M ( b 0 )", "M ( )"};
    const string testTreeCode_M_empty_brackets1[ncodes] = {"", "", "M ( b 1 ) { M ( ) , N ( ) }", "M ( )"};
    const string testTreeCode_M1_empty_brackets1[ncodes] = {"", "", "M ( b 0 )", ""};
    const string testTreeCode_1_empty_brackets1[ncodes] = {"", "", "N ( )", ""};

    const string testMemberTreeCode_empty_brackets2[ncodes] = {"", "", "M ( b 1 ) { M { } , N { } } , N { } , M ( b 0 ) { }", "M { }"};
    const string testTreeCode_M_empty_brackets2[ncodes] = {"", "", "M ( b 1 ) { M { } , N { } }", "M { }"};
    const string testTreeCode_M1_empty_brackets2[ncodes] = {"", "", "M ( b 0 ) { }", ""};
    const string testTreeCode_1_empty_brackets2[ncodes] = {"", "", "N { }", ""};

    const string testMemberTreeCode_empty_brackets3[ncodes] = {"", "", "M ( b 1 ) { M ( ) { } , N ( ) { } } , N ( ) { } , M ( b 0 ) { }", "M ( ) { }"};
    const string testTreeCode_M_empty_brackets3[ncodes] = {"", "", "M ( b 1 ) { M ( ) { } , N ( ) { } }", "M ( ) { }"};
    const string testTreeCode_M1_empty_brackets3[ncodes] = {"", "", "M ( b 0 ) { }", ""};
    const string testTreeCode_1_empty_brackets3[ncodes] = {"", "", "N ( ) { }", ""};

    const string testDropParams[ncodes] = {"A", "B", "C { M { M , N } , N , M }", "D { M }"};
    const string testDropParams_empty_brackets[ncodes] = {"A { }", "B { }", "C { M { M { } , N { } } , N { } , M { } }", "D { M { } }"};

    const string testDropMembers[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C", "D ( s N , i 1 )"};
    const string testDropMembers_empty_brackets[ncodes] = {"A ( )", "B ( f 0.1 , i 2 )", "C ( )", "D ( s N , i 1 )"};

    const string testDropMembers_lvl2[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C { M ( b 1 ) , N , M ( b 0 ) }", "D ( s N , i 1 ) { M }"};
    const string testDropMembers_lvl2_empty_brackets1[ncodes] = {"A ( )", "B ( f 0.1 , i 2 )", "C ( ) { M ( b 1 ) , N ( ) , M ( b 0 ) }", "D ( s N , i 1 ) { M ( ) }"};
    const string testDropMembers_lvl2_empty_brackets2[ncodes] = {"A { }", "B ( f 0.1 , i 2 ) { }", "C { M ( b 1 ) , N , M ( b 0 ) }", "D ( s N , i 1 ) { M }"};
    const string testDropMembers_lvl2_empty_brackets3[ncodes] = {"A ( ) { }", "B ( f 0.1 , i 2 ) { }", "C ( ) { M ( b 1 ) , N ( ) , M ( b 0 ) }", "D ( s N , i 1 ) { M ( ) }"};


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

    vector<const string *> testTreeCode_1_vec;
    testTreeCode_1_vec.push_back(&testTreeCode_1[0]);
    testTreeCode_1_vec.push_back(&testTreeCode_1_empty_brackets1[0]);
    testTreeCode_1_vec.push_back(&testTreeCode_1_empty_brackets2[0]);
    testTreeCode_1_vec.push_back(&testTreeCode_1_empty_brackets3[0]);

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

    // count comparison
    const int testCountNParams[ncodes] = {0, 2, 0, 2};
    const int testCountTreeNParams[ncodes] = {0, 2, 2, 2};
    const int testCountDirectNMembers[ncodes] = {0, 0, 3, 1};
    const int testCountTreeNMembers[ncodes] = {0, 0, 5, 1};

    // empty, single, multi element test vectors
    const vector<string> empty_vec;
    const vector<string> single_vec = {"A"};
    const vector<string> multi_vec = {"A", "B", "C"};

    // params code with params of all usual types and the corresponding parameters with different initialization
    const string fparam = "f 1";
    const string iparam = "i 2";
    const string bparam = "b 1";
    const string sparam = "s N";
    const string allParams = fparam + " , " + iparam + " , " + bparam + " , " + sparam;

    double f = 0., f_test = 1.;
    int i = 0, i_test = 2;
    bool b = false, b_test = true;
    string s, s_test = "N";


    // execute tests

    string str; // for storing strings
    int counter; // for storing counters
    int it = 0;
    for (const string * testArray : testTreeCode_vec) {
        // go through different levels of useless brackets use

        for (int j = 0; j < ncodes; ++j) {
            // and through different codes

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

            //cout << readTreeCode(str, 0, "M")  << endl;
            //cout << testTreeCode_M_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, 0, "M") == testTreeCode_M_vec[it][j]);

            //cout << readTreeCode(str, 1, "M")  << endl;
            //cout << testTreeCode_M1_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, 1, "M") == testTreeCode_M1_vec[it][j]);

            //cout << readTreeCode(str, 1)  << endl;
            //cout << testTreeCode_1_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, 1) == testTreeCode_1_vec[it][j]);


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
        }
        //cout << "----------------------------------------------------" << endl << endl;
        ++it;
    }

    // --- COMPOSERS
    //cout << "Composers:" << endl << endl;


    // composeCodes

    //cout << composeCodes("", "") << endl;
    //cout << ""  << endl << endl;
    assert(composeCodes("", "").empty());

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
    assert(composeCodeList(empty_vec).empty());

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
