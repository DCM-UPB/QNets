#ifndef STRING_CODE_UTILITIES
#define STRING_CODE_UTILITIES

#include <string>
#include <sstream>
#include <vector>
#include <limits>

/*
--- String code system of StringCodeComponents ---

Types of string codes:

  idCode             -   a short identifier for object type, e.g. "nnu" for NeuralNetworkUnit.
  params             -   a space and comma separated list string of "name value" pairs, e.g. "i 10 , f 0.3 , b 1" for parameters of basic datatype.
  fullCode           -   "idCode ( params )", where idCode and params are like above.

  memberTreeCode     -   a list string of member ids, composed recursively, e.g. "A ( i 10 ) , B { C ( f 0.3 ) }" ,
                         where A and B are members of the class and C is a member of B and A. Here A and C have a parameter and B has none.

  treeCode           -   "fullCode { memberTreeCode }" e.g. "foo ( i 10 , f 0.3 , b 1 ) { A ( i 10 ) , B { C ( f 0.3 ) } }"

  paramIdCode, memberIdCore - These appear as arguments in functions below and mean the identifiers of parameters or members, respectively.

NOTE 1: You must not forget to put a space between everything! Extra spaces however don't hurt, will be dropped in results of string functions though.
NOTE 2: The idCode identifiers should uniquely identify a certain type among all derived types of StringCodeComponent.
NOTE 3: However you may have multiple codes of the same type / identifier in a list (so use the index argument to distinguish).
NOTE 4: The parameter identifiers of a class and its' derived types must be unique (just as the actual parameter names in code), so a params code list will always have unique element identifiers.
NOTE 5: You do not have to leave out empty brackets ( ) or { } like in the examples above. Passing empty brackets is completely legal (just as passing empty codes, which then always yields empty function results).
NOTE 6: Unfortunately parameters of string type must not contain any spaces, commas, or brackets of type () or {}.
*/

// --- Readers

std::string readIdCode(const std::string &fullCode); // read idCode string from fullCode or treeCode
std::string readParams(const std::string &fullCode); // read params string from fullCode or treeCode
std::string readParamValue(const std::string &params, const std::string &paramIdCode); // return the value string of certain paramId
std::string readMemberTreeCode(const std::string &treeCode); // return a list string composed of the treeCodes of all members in treeCode
std::string readTreeCode(const std::string &memberTreeCode, const int &index, const std::string &memberIdCode = ""); // return the treeCode of the '(index-1)'th member (first member per default) from memberTreeCode (if passed, only those with matching memberIdCode identifier)
//std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode){return readTreeCode(memberTreeCode, 0, memberIdCode);} // return the treeCode of the first member with matching memberIdCode identifier from memberTreeCode
//std::string readTreeCode(const std::string &memberTreeCode, const int &index){return readTreeCode(memberTreeCode, index, "");}
// --- Drop

std::string dropParams(const std::string &code); // returns the a copy of the same code with all params dropped (i.e. only idCodes)
std::string dropMembers(const std::string &code, const int &drop_lvl = 1); // returns the a copy of the same code with all members after tree level lvl dropped (i.e. lvl==1 -> no members at all)

// --- Counters

int countNParams(const std::string &params); // count number of params in params string
int countTreeNParams(const std::string &treeCode); // count total number of params in treeCode string
int countNMembers(const std::string &memberTreeCode, const bool &direct_only = true); // count number of direct (or total if direct_only == false) members in memberTreeCode string

// --- Composers

std::string composeCodes(const std::string &code1, const std::string &code2); // compose a string of the two codes separated by spaces and comma
std::string composeCodeList(const std::vector<std::string> &codes); // compose a string of vector elements separated by spaces and comma
std::string composeFullCode(const std::string &idCode, const std::string &params); // compose fullCode string from idCode and params
std::string composeTreeCode(const std::string &fullcode, const std::string &memberTreeCode); // compose treeCode string from fullCode and memberTreeCode


// --- Templates

// for applying parameter value string to actual parameter, from param value string
template <typename T>
bool setParamValue(const std::string &paramValue, T &var)
{
    if (paramValue != "") {std::istringstream iss(paramValue); return !(iss >> var).fail();}
    else return false;
}

// for applying parameter value string to actual parameter, from full params list
template <typename T>
bool setParamValue(const std::string &params, const std::string &paramIdCode, T &var)
{
    return setParamValue(readParamValue(params, paramIdCode), var);
}

// for creating parameter value string from actual parameter
template <typename T>
std::string composeParamValue(const T &var)
{
    std::ostringstream oss;
    int p = std::numeric_limits<T>::max_digits10;
    oss.precision(p);
    if (!(oss << var).fail()) return oss.str();
    else return "";
}

// for creating "name value" string from identifier and actual parameter
template <typename T>
std::string composeParamCode(const std::string &paramIdCode, const T &var)
{
    std::string value = composeParamValue(var);
    if (value != "") return paramIdCode + " " + value;
    else return "";
}

#endif

