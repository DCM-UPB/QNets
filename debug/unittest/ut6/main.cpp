#include "StringCodeUtilities.hpp"

#include <iostream>
#include <vector>
#include <assert.h>


int main(){
    using namespace std;

    static const int ncodes = 4;

    // These codes will be used as input for string code methods

    static const string testTreeFullCode[ncodes] = {"A", "B ( f 0.1 , i 2 )", "C { M ( b 1 ) { MM } , N , M ( b 0 ) }", "D ( s name ) { M }"};
    static const string testTreeFullCode_useless_brackets1[ncodes] = {"A ( )" , "B ( f 0.1 , i 2 )" , "C ( ) { M ( b 1 ) { MM ( ) } , N ( ) , M ( b 0 ) }", "D ( s name ) { M ( ) }"};
    static const string testTreeFullCode_useless_brackets2[ncodes] = {"A { }" , "B ( f 0.1 , i 2 ) { }" , "C { M ( b 1 ) { MM { } } , N { } , M ( b 0 ) { } }", "D ( s name ) { M { } }"};
    static const string testTreeFullCode_useless_brackets3[ncodes] = {"A ( ) { }" , "B ( f 0.1 , i 2 ) { }" , "C ( ) { M ( b 1 ) { MM ( ) { } } , N ( ) { } , M ( b 0 ) { } }", "D ( s name ) { M ( ) { } }"};

    static const string testTreeIdCode[ncodes] = {"A", "B", "C { M { MM } , N , M }", "D { M }"};
    static const string testTreeIdCode_useless_brackets[ncodes] = {"A { }", "B { }", "C { M { MM { } } , N { } , M { } }", "D { M { } }"};

    // these codes will be used both as input and as test

    static const string testIdCode[ncodes] = {"A", "B", "C", "D"};
    static const string testParams[ncodes] = {"", "f 0.1 , i 2", "", "s name"};
    static const string testParamValue_i[ncodes] = {"", "2", "", ""};

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


    std::string str; // for storing strings
    int counter; // for storing counters

    for (int i=0; i<ncodes; ++i) {

        // fullCodes
        //cout << "fullCodes:" << endl;
        int it = 0;
        for ( const string * testArray : testTreeFullCode_vec ) {

            // --- READERS

            // readIdCode
            //cout << readIdCode(testArray[i]) << endl;
            //cout << testIdCode[i] << endl << endl;
            assert(readIdCode(testArray[i]) == testIdCode[i]);

            // readParams / readParamValue
            str = readParams(testArray[i]);
            //cout << str << endl;
            //cout << testParams[i] << endl << endl;
            assert(str == testParams[i]);

            //cout << readParamValue(str, "i") << endl;
            //cout << testParamValue_i[i] << endl << endl;
            assert(readParamValue(str, "i") == testParamValue_i[i]);

            // read(Member)TreeCode
            str = readMemberTreeCode(testArray[i]);
            //cout << str  << endl;
            //cout << testMemberTreeFullCode_vec[it][i]  << endl << endl;

            assert(str == testMemberTreeFullCode_vec[it][i]);

            //cout << readTreeCode(str, "M")  << endl;
            //cout << testTreeFullCode_M_vec[it][i]  << endl << endl;
            assert(readTreeCode(str, "M") == testTreeFullCode_M_vec[it][i]);

            //cout << readTreeCode(str, "M", 1)  << endl;
            //cout << testTreeFullCode_M1_vec[it][i]  << endl << endl;
            assert(readTreeCode(str, "M", 1) == testTreeFullCode_M1_vec[it][i]);


            // --- COUNTERS

            // countNParams
            counter = countNParams(readParams(testArray[i]));
            //cout << counter << " <-> " << testCountNParams[i] << endl << endl;
            assert(counter == testCountNParams[i]);

            // countTreeNParams
            counter = countTreeNParams(testArray[i]);
            //cout << counter << " <-> " << testCountTreeNParams[i] << endl << endl;
            assert(counter == testCountTreeNParams[i]);

            // countNMembers (direct)
            str = readMemberTreeCode(testArray[i]);
            counter = countNMembers(str); // default is direct_only=true
            //cout << counter << " <-> " << testCountDirectNMembers[i] << endl << endl;
            assert(counter == testCountDirectNMembers[i]);

            // countNMembers (tree)
            counter = countNMembers(str, false); // count through tree
            //cout << counter << " <-> " << testCountTreeNMembers[i] << endl << endl;
            assert(counter == testCountTreeNMembers[i]);


            ++it;
        }
        //cout << endl;

        // idCodes
        //cout << "idCodes:" << endl;
        it = 0;
        for ( const string * testArray : testTreeIdCode_vec ) {

            // --- READERS

            // readIdCode
            assert(readIdCode(testArray[i]) == testIdCode[i]);

            // read(Member)TreeCode
            str = readMemberTreeCode(testArray[i]);
            //cout << str   << endl;
            //cout << testMemberTreeIdCode_vec[it][i]  << endl << endl;
            assert(str == testMemberTreeIdCode_vec[it][i]);

            //cout << readTreeCode(str, "M") << endl;
            //cout << testTreeIdCode_M_vec[it][i]  << endl << endl;
            assert(readTreeCode(str, "M") == testTreeIdCode_M_vec[it][i]);

            //cout << readTreeCode(str, "M", 1) << endl;
            //cout << testTreeIdCode_M1_vec[it][i] << endl << endl;
            assert(readTreeCode(str, "M", 1) == testTreeIdCode_M1_vec[it][i]);


            // --- COUNTERS

            // countNMembers (direct)
            str = readMemberTreeCode(testArray[i]);
            counter = countNMembers(str); // default is direct_only=true
            //cout << counter << " <-> " << testCountDirectNMembers[i] << endl << endl;
            assert(counter == testCountDirectNMembers[i]);

            // countNMembers (tree)
            counter = countNMembers(str, false); // count through tree
            //cout << counter << " <-> " << testCountTreeNMembers[i] << endl << endl;
            assert(counter == testCountTreeNMembers[i]);


            ++it;
        }

        //cout << "----------------------------------------------------" << endl << endl;
    }


    return 0;
}
