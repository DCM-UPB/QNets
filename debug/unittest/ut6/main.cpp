#include "StringCodeUtilities.hpp"

#include <iostream>
#include <vector>
#include <assert.h>


int main(){
    using namespace std;

    static const int ncodes = 4;

    // These codes will be used as test input for string code methods

    static const string testTreeFullCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C { M ( b 1 ) { MM } , N , M ( b 0 ) }", "D ( s name ) { M }"};
    static const string testTreeFullCode_useless_brackets1[ncodes] = {"A ( )" , "B ( f 0.1 , i 2 )" , "C ( ) { M ( b 1 ) { MM ( ) } , N ( ) , M ( b 0 ) }", "D ( s name ) { M ( ) }"};
    static const string testTreeFullCode_useless_brackets2[ncodes] = {"A { }" , "B ( f 0.1 , i 2 ) { }" , "C { M ( b 1 ) { MM { } } , N { } , M ( b 0 ) { } }", "D ( s name ) { M { } }"};
    static const string testTreeFullCode_useless_brackets3[ncodes] = {"A ( ) { }" , "B ( f 0.1 , i 2 ) { }" , "C ( ) { M ( b 1 ) { MM ( ) { } } , N ( ) { } , M ( b 0 ) { } }", "D ( s name ) { M ( ) { } }"};

    static const string testTreeIdCode[ncodes] = {"A", "B", "C { M { MM } , N , M }", "D { M }"};
    static const string testTreeIdCode_useless_brackets[ncodes] = {"A { }", "B { }", "C { M { MM { } } , N { } , M { } }", "D { M { } }"};

    // these codes will be used as comparison (and in the end as input as well)

    static const string testIdCode[ncodes] = {"A", "B", "C", "D"};
    static const string testParams[ncodes] = {"", "f 0.1 , i 2", "", "s name"};
    static const string testParamValue_i[ncodes] = {"", "2", "", ""};
    static const string testFullCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C", "D ( s name )"};

    static const string testMemberTreeFullCode[ncodes] = {"", "", "M ( b 1 ) { MM } , N , M ( b 0 )", "M"};
    static const string testTreeFullCode_M[ncodes] = {"", "", "M ( b 1 ) { MM }", "M"};
    static const string testTreeFullCode_M1[ncodes] = {"", "", "M ( b 0 )", ""};

    static const string testMemberTreeFullCode_useless_brackets1[ncodes] = {"", "", "M ( b 1 ) { MM ( ) } , N ( ) , M ( b 0 )", "M ( )"};
    static const string testTreeFullCode_M_useless_brackets1[ncodes] = {"", "", "M ( b 1 ) { MM ( ) }", "M ( )"};
    static const string testTreeFullCode_M1_useless_brackets1[ncodes] = {"", "", "M ( b 0 )", ""};

    static const string testMemberTreeFullCode_useless_brackets2[ncodes] = {"", "", "M ( b 1 ) { MM { } } , N { } , M ( b 0 ) { }", "M { }"};
    static const string testTreeFullCode_M_useless_brackets2[ncodes] = {"", "", "M ( b 1 ) { MM { } }", "M { }"};
    static const string testTreeFullCode_M1_useless_brackets2[ncodes] = {"", "", "M ( b 0 ) { }", ""};

    static const string testMemberTreeFullCode_useless_brackets3[ncodes] = {"", "", "M ( b 1 ) { MM ( ) { } } , N ( ) { } , M ( b 0 ) { }", "M ( ) { }"};
    static const string testTreeFullCode_M_useless_brackets3[ncodes] = {"", "", "M ( b 1 ) { MM ( ) { } }", "M ( ) { }"};
    static const string testTreeFullCode_M1_useless_brackets3[ncodes] = {"", "", "M ( b 0 ) { }", ""};

    static const string testMemberTreeIdCode[ncodes] = {"", "", "M { MM } , N , M", "M"};
    static const string testTreeIdCode_M[ncodes] = {"", "", "M { MM }", "M"};
    static const string testTreeIdCode_M1[ncodes] = {"", "", "M", ""};

    static const string testMemberTreeIdCode_useless_brackets[ncodes] = {"", "", "M { MM { } } , N { } , M { }", "M { }"};
    static const string testTreeIdCode_M_useless_brackets[ncodes] = {"", "", "M { MM { } }", "M { }"};
    static const string testTreeIdCode_M1_useless_brackets[ncodes] = {"", "", "M { }", ""};

    static const int testCountNParams[ncodes] = {0, 2, 0, 1};
    static const int testCountTreeNParams[ncodes] = {0, 2, 2, 1};
    static const int testCountDirectNMembers[ncodes] = {0, 0, 3, 1};
    static const int testCountTreeNMembers[ncodes] = {0, 0, 4, 1};


    // create input treecode vectors
    vector<const string *> testTreeFullCode_vec;
    testTreeFullCode_vec.push_back(&testTreeFullCode[0]);
    testTreeFullCode_vec.push_back(&testTreeFullCode_useless_brackets1[0]);
    testTreeFullCode_vec.push_back(&testTreeFullCode_useless_brackets2[0]);
    testTreeFullCode_vec.push_back(&testTreeFullCode_useless_brackets3[0]);

    vector<const string *> testTreeIdCode_vec;
    testTreeIdCode_vec.push_back(&testTreeIdCode[0]);
    testTreeIdCode_vec.push_back(&testTreeIdCode_useless_brackets[0]);

    // create comparison vectors
    vector<const string *> testMemberTreeFullCode_vec;
    testMemberTreeFullCode_vec.push_back(&testMemberTreeFullCode[0]);
    testMemberTreeFullCode_vec.push_back(&testMemberTreeFullCode_useless_brackets1[0]);
    testMemberTreeFullCode_vec.push_back(&testMemberTreeFullCode_useless_brackets2[0]);
    testMemberTreeFullCode_vec.push_back(&testMemberTreeFullCode_useless_brackets3[0]);

    vector<const string *> testTreeFullCode_M_vec;
    testTreeFullCode_M_vec.push_back(&testTreeFullCode_M[0]);
    testTreeFullCode_M_vec.push_back(&testTreeFullCode_M_useless_brackets1[0]);
    testTreeFullCode_M_vec.push_back(&testTreeFullCode_M_useless_brackets2[0]);
    testTreeFullCode_M_vec.push_back(&testTreeFullCode_M_useless_brackets3[0]);

    vector<const string *> testTreeFullCode_M1_vec;
    testTreeFullCode_M1_vec.push_back(&testTreeFullCode_M1[0]);
    testTreeFullCode_M1_vec.push_back(&testTreeFullCode_M1_useless_brackets1[0]);
    testTreeFullCode_M1_vec.push_back(&testTreeFullCode_M1_useless_brackets2[0]);
    testTreeFullCode_M1_vec.push_back(&testTreeFullCode_M1_useless_brackets3[0]);

    vector<const string *> testMemberTreeIdCode_vec;
    testMemberTreeIdCode_vec.push_back(&testMemberTreeIdCode[0]);
    testMemberTreeIdCode_vec.push_back(&testMemberTreeIdCode_useless_brackets[0]);

    vector<const string *> testTreeIdCode_M_vec;
    testTreeIdCode_M_vec.push_back(&testTreeIdCode_M[0]);
    testTreeIdCode_M_vec.push_back(&testTreeIdCode_M_useless_brackets[0]);

    vector<const string *> testTreeIdCode_M1_vec;
    testTreeIdCode_M1_vec.push_back(&testTreeIdCode_M1[0]);
    testTreeIdCode_M1_vec.push_back(&testTreeIdCode_M1_useless_brackets[0]);


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

        // fullCodes
        //cout << "fullCodes:" << endl << endl;
        int it = 0;
        for ( const string * testArray : testTreeFullCode_vec ) {

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
            //cout << testMemberTreeFullCode_vec[it][j]  << endl << endl;

            assert(str == testMemberTreeFullCode_vec[it][j]);

            //cout << readTreeCode(str, "M")  << endl;
            //cout << testTreeFullCode_M_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, "M") == testTreeFullCode_M_vec[it][j]);

            //cout << readTreeCode(str, "M", 1)  << endl;
            //cout << testTreeFullCode_M1_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, "M", 1) == testTreeFullCode_M1_vec[it][j]);


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
        //cout << endl;

        // idCodes
        //cout << "idCodes:" << endl << endl;
        it = 0;
        for ( const string * testArray : testTreeIdCode_vec ) {

            // --- READERS

            // readIdCode
            assert(readIdCode(testArray[j]) == testIdCode[j]);

            // read(Member)TreeCode
            str = readMemberTreeCode(testArray[j]);
            //cout << str   << endl;
            //cout << testMemberTreeIdCode_vec[it][j]  << endl << endl;
            assert(str == testMemberTreeIdCode_vec[it][j]);

            //cout << readTreeCode(str, "M") << endl;
            //cout << testTreeIdCode_M_vec[it][j]  << endl << endl;
            assert(readTreeCode(str, "M") == testTreeIdCode_M_vec[it][j]);

            //cout << readTreeCode(str, "M", 1) << endl;
            //cout << testTreeIdCode_M1_vec[it][j] << endl << endl;
            assert(readTreeCode(str, "M", 1) == testTreeIdCode_M1_vec[it][j]);


            // --- COUNTERS

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

    //cout << composeTreeCode(testIdCode[0], testMemberTreeIdCode[0]) << endl;
    //cout << testTreeIdCode[0] << endl << endl;
    assert(composeTreeCode(testIdCode[0], testMemberTreeIdCode[0]) == testTreeIdCode[0]); // empty members

    //cout << composeTreeCode(testIdCode[3], testMemberTreeIdCode[3]) << endl;
    //cout << testTreeIdCode[3] << endl << endl;
    assert(composeTreeCode(testIdCode[3], testMemberTreeIdCode[3]) == testTreeIdCode[3]); // with member


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
