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
            if (word!=",") return word; // in case paramValue was not empty
            else break;
        }
    }
    return "";
}


// readMemberTreeCode

void readMemberTreeCode(std::istringstream &iss, std::string &memberTreeCode, const int &drop_lvl = 0) // internal helper
{
    std::string word;
    int countOpenBrackets = 1; // count total open { brackets, assuming the first one is already skipped

    while (iss >> word) { // read in memberTreeCodes
        if (word == "{") ++countOpenBrackets;
        if (drop_lvl > 0 && countOpenBrackets>=drop_lvl) {
            if (word == "}") --countOpenBrackets;
            continue; // drop members past level drop_lvl, i.e. drop_lvl==1 yields empty MemberTreeCode
        }
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
    int countOpenRoundBrackets = 0;
    int countLeftBrackets = 0; // count { brackets of member code
    int countRightBrackets = 0; // count } brackets of member code

    // assuming the memberIdCode is already found and written into treeCode
    while (iss >> word) { // read rest
        if (countLeftBrackets == 0 && countOpenRoundBrackets == 0 && word == ",") return; // there was no members list
        if (word == "(") ++countOpenRoundBrackets;
        if (word == ")") --countOpenRoundBrackets;
        if (word == "{") ++countLeftBrackets;
        if (word == "}") ++countRightBrackets;
        treeCode += " " + word;
        if (countLeftBrackets > 0 && countLeftBrackets == countRightBrackets) return; // done
    }
    return;
}


std::string readTreeCode(const std::string &memberTreeCode, const int &index, const std::string &memberIdCode) // public function
{
    std::istringstream iss(memberTreeCode);
    std::string word;
    std::string treeCode = "";
    int countIndex = 0;
    int countOpenBrackets1 = 0;
    int countOpenBrackets2 = 0;

    while (iss >> word) { // search for memberIdCode
        if (word == "{") ++countOpenBrackets1;
        if (word == "}") --countOpenBrackets1;
        if (word == "(") ++countOpenBrackets2;
        if (word == ")") --countOpenBrackets2;
        if (countOpenBrackets1 == 0 && countOpenBrackets2 == 0) { // make sure we count nothing inside brackets
            if (memberIdCode == "") {
                if (word == ",") {++countIndex; continue;} // count commas in this case
                if (countIndex == index){
                    treeCode = word; // read IdCode
                    readTreeCode(iss, treeCode); // read the rest of the treeCode
                    return treeCode;
                }
            }
            else if (word == memberIdCode) {
                if (countIndex < index) {++countIndex; continue;} // count id appearances in this case
                else {
                    treeCode = word; // read IdCode
                    readTreeCode(iss, treeCode); // read the rest of the treeCode
                    return treeCode;
                }
            }
        }
    }
    return treeCode;
}


// --- Drop


// dropParams

std::string dropParams(const std::string &code)
{
    std::istringstream iss(code);
    std::string word;
    std::string newCode = "";

    while (iss >> word) {
        if (word == "(") {readParams(iss, word); continue;} // skip any params
        if (newCode != "") newCode += " "; // add spacing
        newCode += word;
    }
    return newCode;
}


// dropMembers

std::string dropMembers(const std::string &code, const int &drop_lvl)
{
    std::istringstream iss(code);
    std::string word;
    std::string newCode = "";

    while (iss >> word) {
        if (word == "{") {
            word = "";
            if (drop_lvl>1) {
                newCode += " { ";
                readMemberTreeCode(iss, word, drop_lvl); // drop lvl drop_lvl and beyond
                if (word!="") newCode += word + " ";
                newCode += "}";
            }
            else readMemberTreeCode(iss, word, 1); // skip all member code
            continue;
        }
        if (newCode != "") newCode += " "; // add spacing
        newCode += word;
    }
    return newCode;
}


// --- Counters


// countNParams

void countNParams(std::istringstream &iss, int &counter) // internal helper
{
    bool foundSomething = false;
    std::string word;

    while(iss >> word) { // assuming a possible opening bracket ( is already skipped
        if (word == ")") break; // actually allows passing arbitrary codes/iss with opened params bracket
        else if (word == ",") ++counter; // count every comma
        else foundSomething = true; // found something except bracket or comma
    }
    if (foundSomething) {++counter;} // if there was something, we need to do the final increment
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
    bool bracketClosed = false; // assuming the first { one is already skipped

    while (iss >> word) { // go through memberTreeCode
        if (word == "(") countNParams(iss, counter); // count params bracket
        else if (word == "{") countMemberNParams(iss, counter); // recursive call
        else if (word == "}") bracketClosed = true;
        if (bracketClosed) return; // done
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
    if (word == "{") {
        while (iss >> word) {
            countMemberNParams(iss, counter); // memberTreeCode counting
        }
    }
    return counter;
}


// countNMembers

void countMemberNMembers(std::istringstream &iss, int &counter) // internal helper
{
    std::string word;
    int dummy_counter = 0;
    bool foundSomething = false;
    bool bracketClosed = false; // assuming the first { one is already skipped/open

    while (iss >> word) { // go through memberTreeCode
        if (word == "(") countNParams(iss, dummy_counter); // skip params brackets
        else if (word == "{") countMemberNMembers(iss, counter); // recursively count members
        else if (word == "}") bracketClosed = true;
        else if (word == ",") ++counter; // count commas
        else foundSomething = true; // found something except bracket or comma
        if (bracketClosed) break; // done
    }
    if (foundSomething) ++counter; // if there was something, we need to do the final increment
    return;
}

int countNMembers(const std::string &memberTreeCode, const bool &direct_only) // public function, count number of direct (or total if direct_only==false) members in memberTreeCode string
{
    std::istringstream iss(memberTreeCode);
    std::string word;
    int counter = 0, dummy_counter = 0;

    while (iss >> word) {
        if (word == "(") countNParams(iss, dummy_counter); // skip params bracket
        if (word == "{") {
            if (direct_only) countMemberNMembers(iss, dummy_counter); // skip member content
            else countMemberNMembers(iss, counter); // count member content
        }
        if (word == ",") ++counter; // count commas
    }
    if (memberTreeCode != "") return counter+1; // we did only count commas, so we have to increment by 1
    else return 0; // except if there was nothing
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
    std::string codeList = "";
    for (std::vector<std::string>::size_type i=0; i<codes.size(); ++i) {
        codeList = composeCodes(codeList, codes[i]);
    }
    return codeList;
}


std::string composeFullCode(const std::string &idCode, const std::string &params)
{
    if (idCode != "" && params != "") return idCode + " ( " + params + " )";
    return idCode;
}


std::string composeTreeCode(const std::string &fullCode, const std::string &memberTreeCode)
{
    if (fullCode != "" && memberTreeCode != "") return fullCode + " { " + memberTreeCode + " }";
    return fullCode;
}
