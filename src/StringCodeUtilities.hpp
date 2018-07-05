#ifndef STRING_CODE_UTILITIES
#define STRING_CODE_UTILITIES

#include <string>
#include <sstream>
#include <vector>
#include <limits>

using namespace std;

/*
--- String code system of SerializableComponents ---

Currently we serialize the configuration of a SerializableComponent into a simple string with our string code system.

The full string code (a.k.a treeCode) of a component looks like the follwing:
"idCode ( params ) { memberTreeCode }" , where:

  idCode             -   a short identifier for object type, e.g. "fbc" for FooBarComponent.
  params             -   a space and comma separated list string of "paramIdCode paramValue" pairs for object members of basic type, e.g. "foo 2 , bar 0.5"
  memberTreeCode     -   a space and comma separated list string of treeCodes of serializable members, composed recursively, e.g. "A ( b 0 ) , B { C ( label foobar ) }" ,
                         where A and B are members of the object and C is a member of B (A and C have no serializable members). A and C have a parameter and B has none.

So if we put idCode and params together we get the so called fullCode:
  fullCode           -   "idCode ( params )", e.g. "fbc ( foo 2 , bar 0.5 )"

If we also add the memberTreeCode, we obtain the so called treeCode:
  treeCode           -   "fullCode { memberTreeCode }" e.g. "fbc ( foo 2 , bar 0.5 ) { A ( flag 0 ), B { C ( label foobar ) } }"

  paramIdCode, memberIdCore - These appear as arguments in functions below and mean the identifiers of parameters or members, respectively.

NOTE 1: You must not forget to put a space between everything! Extra spaces however don't hurt, will be dropped in results of string functions though.
NOTE 2: The idCode identifiers should uniquely identify a certain type among all derived types of SerializableComponent.
NOTE 3: However you may have multiple codes of the same type / identifier in a list (so use the index argument to distinguish).
NOTE 4: The parameter identifiers of a class and its' derived types must be unique (just as the actual parameter names in code), so a params code list will always have unique element identifiers.
NOTE 5: You do not have to leave out empty brackets ( ) or { } like in the examples above. Passing empty brackets is completely legal (just as passing empty codes, which then always yields empty function results).
NOTE 6: Unfortunately parameters of string type must not contain any spaces, commas, or brackets of type () or {}.
*/

// --- Readers

string readIdCode(const string &fullCode); // read idCode string from fullCode or treeCode
string readParams(const string &fullCode); // read params string from fullCode or treeCode
string readParamValue(const string &params, const string &paramIdCode); // return the value string of certain paramId
string readMemberTreeCode(const string &treeCode); // return a list string composed of the treeCodes of all members in treeCode
string readTreeCode(const string &memberTreeCode, const int &index, const string &memberIdCode = ""); // return the treeCode of the '(index-1)'th member (first member per default) from memberTreeCode (if passed, only those with matching memberIdCode identifier)

// --- Drop

string dropParams(const string &code); // returns the a copy of the same code with all params dropped (i.e. only idCodes)
string dropMembers(const string &code, const int &drop_lvl = 1); // returns the a copy of the same code with all members after tree level lvl dropped (i.e. lvl==1 -> no members at all)

// --- Counters

int countNParams(const string &params); // count number of params in params string
int countTreeNParams(const string &treeCode); // count total number of params in treeCode string
int countNMembers(const string &memberTreeCode, const bool &direct_only = true); // count number of direct (or total if direct_only == false) members in memberTreeCode string

// --- Composers

string composeCodes(const string &code1, const string &code2); // compose a string of the two codes separated by spaces and comma
string composeCodeList(const vector<string> &codes); // compose a string of vector elements separated by spaces and comma
string composeFullCode(const string &idCode, const string &params); // compose fullCode string from idCode and params
string composeTreeCode(const string &fullcode, const string &memberTreeCode); // compose treeCode string from fullCode and memberTreeCode


// --- Templates

// for applying parameter value string to actual parameter, from param value string
template <typename T>
bool setParamValue(const string &paramValue, T &var)
{
    if (paramValue != "") {istringstream iss(paramValue); return !(iss >> var).fail();}
    else return false;
}

// for applying parameter value string to actual parameter, from full params list
template <typename T>
bool setParamValue(const string &params, const string &paramIdCode, T &var)
{
    return setParamValue(readParamValue(params, paramIdCode), var);
}

// for creating parameter value string from actual parameter
template <typename T>
string composeParamValue(const T &var)
{
    ostringstream oss;
    int p = numeric_limits<T>::max_digits10;
    oss.precision(p);
    if (!(oss << var).fail()) return oss.str();
    else return "";
}

// for creating "name value" string from identifier and actual parameter
template <typename T>
string composeParamCode(const string &paramIdCode, const T &var)
{
    string value = composeParamValue(var);
    if (value != "") return paramIdCode + " " + value;
    else return "";
}

#endif

