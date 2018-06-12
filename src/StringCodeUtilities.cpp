#include "StringCodeUtilities.hpp"

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

// --- Readers

// readIdCode

std::string readIdCode(const std::string &fullCode)
{
    std::istringstream iss(fullCode);
    std::string idCode = "";
    iss >> idCode;
    return idCode;
}


// readParams

void readParams(std::istringstream &iss, std::string &params) // internal helper
{
    std::string word;
    while (iss >> word) { // read params string, assuming opening bracket is already skipped
        if (word == ")") return; // don't add the ) and return
        if (params != "") params += " "; // add spacing
        params += word;
    }
    return; // no closing params bracket found / empty
}

std::string readParams(const std::string &fullCode) // public function (is in header)
{
    std::istringstream iss(fullCode);
    std::string word;
    std::string params = "";

    iss >> word; // skip id
    iss >> word; // skip (
    if (word == "(") readParams(iss, params); // params bracket read
    return params;
}


// readParamValue

std::string readParamValue(const std::string &params, const std::string &paramIdCode)
{
    std::istringstream iss(params);
    std::string word;
    while (iss >> word) { // search for paramIdCode
        if (word == paramIdCode) {
            iss >> word; // read value
            return word;
        }
    }
    return "";
}


// readMemberTreeCode

void readMemberTreeCode(std::istringstream &iss, std::string &memberTreeCode) // internal helper
{
    std::string word;
    int countOpenBrackets = 1; // count total open { brackets, assuming the first one is already skipped

    while (iss >> word) { // read in membersTreeFullCodes
        if (word == "{") ++countOpenBrackets;
        if (word == "}") --countOpenBrackets;
        if (countOpenBrackets == 0) return;
        if (memberTreeCode != "") memberTreeCode += " "; // add spacing
        memberTreeCode += word; // by placing it here the final } wont be added
    }
    return;
}

std::string readMemberTreeCode(const std::string &treeCode) // public function
{
    std::istringstream iss(treeCode);
    std::string word;
    std::string memberTreeCode = "";

    iss >> word; // skip id
    iss >> word; // skip ( or {
    if (word == "(") {
        readParams(iss, word); // skip params bracket
        iss >> word;
    }
    if (word == "{") readMemberTreeCode(iss, memberTreeCode); // memberTreeCode read
    return memberTreeCode;
}


// readTreeCode

void readTreeCode(std::istringstream &iss, std::string &treeCode) // internal helper
{
    std::string word;
    int countLeftBrackets = 0; // count { brackets of member code
    int countRightBrackets = 0; // count } brackets of member code

    // assuming the memberIdCode is already found and written into treeCode
    while (iss >> word) { // read rest
        if (countLeftBrackets == 0 && word == ",") return; // there was no members list
        if (word == "{") ++countLeftBrackets;
        if (word == "}") ++countRightBrackets;
        treeCode += " " + word;
        if (countLeftBrackets > 0 && countLeftBrackets == countRightBrackets) return; // done
    }
    return;
}

std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode) // public function
{
    std::istringstream iss(memberTreeCode);
    std::string word;
    std::string treeCode = "";

    while (iss >> word) { // search for memberIdCode
        if (word == memberIdCode) {
            treeCode = word; // read IdCode
            readTreeCode(iss, treeCode); // read the rest of the treeCode
            return treeCode;
        }
    }
    return treeCode;
}


std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode, const int &index) // public function
{
    std::istringstream iss(memberTreeCode);
    std::string word;
    std::string treeCode = "";
    int countIndex = -1;

    while (iss >> word) { // search for memberIdCode
        if (word == memberIdCode) {
            countIndex += 1; // count until correct index
            if (countIndex == index) {
                treeCode = word; // read IdCode
                readTreeCode(iss, treeCode); // read the rest of the treeCode
                return treeCode;
            }
        }
    }
    return treeCode;
}

// --- Counters


// countNParams

void countNParams(std::istringstream &iss, int &counter) // internal helper
{
    std::string word;

    while(iss >> word) { // assuming a possible opening bracket ( is already skipped
        if (word == ")") {++counter; return;} // last increment, return
        if (word == ",") ++counter; // count every comma
    }
    return;
}

int countNParams(const std::string &params) // public function, count number of params in params code
{
    std::istringstream iss(params);
    std::string word;
    int counter = 0;
    countNParams(iss, counter);
    return counter;
}


// countTreeNParams

void countMemberNParams(std::istringstream &iss, int &counter) // internal helper
{
    std::string word;
    int countOpenBrackets = 1; // count total open { brackets, assuming the first one is already skipped

    while (iss >> word) { // go through membersTreeCodes
        if (word == "{") ++countOpenBrackets;
        if (word == "}") --countOpenBrackets;
        if (countOpenBrackets == 0) return;
        if (word == "(") countNParams(iss, counter);
    }
    return;

}

int countTreeNParams(const std::string &treeCode) // public function, count total number of params in treeCode string
{
    std::istringstream iss(treeCode);
    std::string word;
    int counter = 0;

    iss >> word; // skip idCode
    iss >> word; // skip ( or {
    if (word == "(") {
        countNParams(iss, counter); // count params bracket
        iss >> word;
    }
    if (word == "{") countMemberNParams(iss, counter); // memberTreeCode counting
    return counter;
}


// countNMembers

void countMemberNMembers(std::istringstream &iss, int &counter) // internal helper
{
    std::string word;
    int countOpenBrackets = 1; // count total open { brackets, assuming the first one is already skipped
    while (iss >> word) { // read in membersTreeFullCodes
        if (word == "{") ++countOpenBrackets;
        if (word == "}") --countOpenBrackets;
        if (countOpenBrackets == 0) {++counter; return;}
        if (word == ",") ++counter; // count commas
    }
    return;
}

int countNMembers(const std::string &memberTreeCode, const bool direct_only) // public function, count number of direct (or total if direct_only==false) members in memberTreeCode string
{
    std::istringstream iss(memberTreeCode);
    std::string word;
    int counter = 0, dummy_counter = 0;


    while (iss >> word) {
        if (word == "{") {
            if (direct_only) countMemberNMembers(iss, dummy_counter); // skip member content
            else countMemberNMembers(iss, counter); // count member content
        }
        if (word == "}") return counter+1; // this is the outer right bracket
        if (word == ",") ++counter; // count commas
    }
    return counter;
}


// --- Composers

std::string composeCodes(const std::string &code1, const std::string &code2)
{
    if (code1=="") return code2;
    if (code2=="") return code1;
    return code1 + " , " + code2;
}


std::string composeCodeList(const std::vector<std::string> &codes)
{
    if (codes.size() == 0) return ""; // nothing to be done
    std::string codeList = codes[0]; // start with first code
    for (std::vector<std::string>::size_type i=1; i<codes.size(); ++i) codeList += " , " + codes[i]; // append other codes with spacing and comma
    return codeList;
}


std::string composeFullCode(const std::string &idCode, const std::string &params)
{
    if (params != "") return idCode + " ( " + params + " )";
    return idCode;
}


std::string composeTreeCode(const std::string &code, const std::string &memberTreeCode)
{
    if (memberTreeCode != "") return code + " { " + memberTreeCode + " }";
    return code;
}
