Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 5 items.

=== ITEM id=abcfc8a3d9bceb6b ===
Gold sentences and facts:
Sentence 0: Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 .
  Fact 0.0: Sheila Carter is a fictional character from The Young and the Restless.
  Fact 0.1: Sheila Carter is a fictional character from The Bold and the Beautiful.
  Fact 0.2: The Young and the Restless and The Bold and the Beautiful are American soap operas on the CBS network.
  Fact 0.3: The character Sheila Carter was created by William J. Bell.
  Fact 0.4: The role of Sheila Carter was introduced in 1990.
  Fact 0.5: The role of Sheila Carter was introduced by Edward J. Scott.
  Fact 0.6: The role of Sheila Carter was introduced under the portrayal of Kimberlin Brown.
  Fact 0.7: Kimberlin Brown portrayed the role of Sheila Carter for many intervals on both soaps until 2006.
  Fact 0.8: In 2006, Michelle Stafford took over the role of Sheila Carter.
  Fact 0.9: Michelle Stafford took over the role after Sheila had plastic surgery to look like Phyllis Summers.
  Fact 0.10: Kimberlin Brown returned to the role of Sheila Carter on The Bold and the Beautiful.
  Fact 0.11: Kimberlin Brown's return to the role of Sheila Carter on The Bold and the Beautiful began on June 9, 2017.
Sentence 1: The role was portrayed by Kimberlin Brown for a total of 16 years . Her first run was on The Young and the Restless from May 1990 to May 20 , 1992 , when the character crossed over to The Bold and the Beautiful from May 21 , 1992 , to October 2 , 1998 , again from May 24 to November 5 , 2002 , and again finally from September 26 to October 10 , 2003 . Despite being a regular cast member on The Bold and the Beautiful , Brown returned to The Young and the Restless for several guest appearances beginning on October 27 , 1992 . Brown departed on November 5 , 1992 , and reappeared for one episode on December 23 , 1992 . She continued making appearance from January 7 to 14 , 1993 , February 19 , 1993 , to March 2 , 1993 , October 11 , 1993 , October 19 to 27 , 1993 , January 12 , 1994 , and February 28 , 1995 , until March 2 , 1995 .
  Fact 1.0: Kimberlin Brown portrayed Sheila Carter for a total of 16 years.
  Fact 1.1: Brown's first run as Sheila Carter was on The Young and the Restless from May 1990 to May 20, 1992.
  Fact 1.2: The character Sheila Carter crossed over to The Bold and the Beautiful from May 21, 1992, to October 2, 1998.
  Fact 1.3: The character Sheila Carter appeared on The Bold and the Beautiful again from May 24 to November 5, 2002.
  Fact 1.4: The character Sheila Carter appeared on The Bold and the Beautiful again finally from September 26 to October 10, 2003.
  Fact 1.5: Brown was a regular cast member on The Bold and the Beautiful.
  Fact 1.6: Despite being a regular cast member on The Bold and the Beautiful, Brown returned to The Young and the Restless for several guest appearances beginning on October 27, 1992.
  Fact 1.7: Brown departed from The Young and the Restless on November 5, 1992.
  Fact 1.8: Brown reappeared on The Young and the Restless for one episode on December 23, 1992.
  Fact 1.9: Brown continued making appearances from January 7 to 14, 1993.
  Fact 1.10: Brown made an appearance on February 19, 1993.
  Fact 1.11: Brown made appearances from February 19, 1993, to March 2, 1993.
  Fact 1.12: Brown made an appearance on October 11, 1993.
  Fact 1.13: Brown made appearances from October 19 to 27, 1993.
  Fact 1.14: Brown made an appearance on January 12, 1994.
  Fact 1.15: Brown made an appearance on February 28, 1995, until March 2, 1995.
Sentence 2: Sheila Carter Kimberlin Brown as Sheila Carter The Young and the Restless character Portrayed by Kimberlin Brown ( 1990 -- ) Michelle Stafford ( 2006 -- 07 ) Duration 1990 -- 98 2002 -- 03 2005 -- 07 2017 -- First appearance May 1990 Created by William J. Bell Introduced by Edward J. Scott ( 1990 , 1992 -- 95 ) Lee Phillip Bell ( 1992 ) Bradley Bell ( 2002 -- 03 ) John F. Smith ( 2005 ) Lynn Marie Latham and Josh Griffith ( 2006 ) Bradley Bell and Michael Minnis ( 2017 ) Crossover appearances The Bold and the Beautiful Classification Present ; regular Profile Other names Sheila Grainger Sheila Forrester Sheila Warwick Sybil Weller Sheila Lovejoy Brenda Harris Jennifer Mitchell Phyllis Newman Betsy Occupation Waitress Executive assistant Nurse Residence Los Angeles , California Michelle Stafford as Sheila Carter ( show ) Family Family Carter Mother Molly Carter Sisters Sarah Smythe Husband Scott Grainger , Sr. ( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter Granddaughters Lucy Romalotti
  Fact 2.0: Sheila Carter is a character portrayed by Kimberlin Brown from 1990 onward.
  Fact 2.1: Sheila Carter is a character portrayed by Michelle Stafford from 2006 to 2007.
  Fact 2.2: The character Sheila Carter's duration on the soaps covers 1990-98, 2002-03, 2005-07 and 2017 onward.
  Fact 2.3: The character Sheila Carter first appeared in May 1990.
  Fact 2.4: The character Sheila Carter was created by William J. Bell.
  Fact 2.5: The character Sheila Carter was introduced by Edward J. Scott in 1990 and 1992-95.
  Fact 2.6: The character Sheila Carter was introduced by Lee Phillip Bell in 1992.
  Fact 2.7: The character Sheila Carter was introduced by Bradley Bell in 2002-03.
  Fact 2.8: The character Sheila Carter was introduced by John F. Smith in 2005.
  Fact 2.9: The character Sheila Carter was introduced by Lynn Marie Latham and Josh Griffith in 2006.
  Fact 2.10: The character Sheila Carter was introduced by Bradley Bell and Michael Minnis in 2017.
  Fact 2.11: The character Sheila Carter has crossover appearances on The Bold and the Beautiful.
  Fact 2.12: The character Sheila Carter is classified as present and regular.
  Fact 2.13: Sheila Carter's other names include Sheila Grainger, Sheila Forrester, Sheila Warwick, Sybil Weller, Sheila Lovejoy, Brenda Harris, Jennifer Mitchell, Phyllis Newman and Betsy.
  Fact 2.14: Sheila Carter's occupations include waitress, executive assistant and nurse.
  Fact 2.15: Sheila Carter's residence is Los Angeles, California.
  Fact 2.16: Sheila Carter's family is the Carter family.
  Fact 2.17: Sheila Carter's mother is Molly Carter.
  Fact 2.18: Sheila Carter's sister is Sarah Smythe.
  Fact 2.19: Sheila Carter's husband was Scott Grainger, Sr. from 1991 to 1992.
  Fact 2.20: Sheila Carter's husband was Eric Forrester from 1993 to 1995.
  Fact 2.21: Sheila Carter's husband was James Warwick from 1997 to 1998.
  Fact 2.22: Sheila Carter's son is Ryder Callahan.
  Fact 2.23: Sheila Carter's daughters are Mary Carter Warwick, Diana Carter and Daisy Carter.
  Fact 2.24: Sheila Carter's granddaughter is Lucy Romalotti.
  Fact 2.25: Michelle Stafford portrayed Sheila Carter on the show.

Retrieved items:
<item 1>
Sheila Carter
When asked where her alter - ego may be `` hiding out '' , Brown said : `` I am still getting healthy from my car accident . Right now , Sheila is hiding out in
North Carolina -- playing golf . But do n't worry : Sheila always comes back ! '' In June 2017 , Kimberlin Brown reprised her role as Sheila on The Bold and the
Beautiful . Executive producer Brad Bell announced Brown had been signed to a one - year deal with the serial . Development ( edit ) `` She seduced another
woman 's husband and stole their child ... but if you think that 's bold , hold on tight ... because the hand that rocked The Young and the Restless is now
rocking The Bold and the Beautiful . Sheila lives ... -- CBS promoting Kimberlin Brown 's crossover as Sheila On the topic of Sheila and Lauren 's `` cat fights
'' , Brown said : `` People like cat fights because we can get away with what you ca n't get away with in everyday life . You 're living vicariously through me
.
</item 1>

<item 2>
Sheila Carter
CBS.com . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Wednesday , 11th January 2006 '' . CBS.com . Retrieved 16 September 2012 .
Jump up ^ `` The Young and the Restless : Thursday , 28th December 2006 '' . CBS.com . Retrieved 16 September 2012 . Jump up ^ `` Michelle Stafford Playing
Sheila Carter on The Young and the Restless '' . She Knows Soaps . Retrieved 16 September 2012 . Jump up ^ Kimberlin Brown ( 2007 ) . KB Interview Nov 2007 .
KOUS News . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday , 9th February 2007 '' . CBS.com . Retrieved 16 September 2012 . Jump
up ^ Kerr , Luke ( 28 August 2009 ) . `` Could Kimberlin Brown Return to B&B ? '' . Daytime Confidential . Retrieved 16 September 2012 .
</item 2>

<item 3>
Sheila Carter
However , that December , the character returned , though not portrayed by Brown , but by Michelle Stafford , who in addition portrayed Phyllis Summers . For
the reason of Brown not returning to the role , the actress said : `` I had a bad accident last year , and when they wanted me back in November , I really could
n't go at that time , and that 's why I came back looking like somebody else . '' Stafford only portrayed the role for two months until February 9 , 2007 , when
Sheila was shot by her longtime enemy Lauren Fenmore and presumed dead . In 2009 , reports speculated that Brown would reprise the role again on The Bold and
the Beautiful , which was followed by more rumors speculating a return to The Young and the Restless . `` I actually got a call from ( B&B head writer and
executive producer ) Brad Bell a while back asking if I 'd be interested in coming back , and I said , ' Sure ! ' The thing with daytime is that the storylines
are ( written ) so far in advance , you just never know when ( you might be needed ) , '' Brown said at the time .
</item 3>

<item 4>
Sheila Carter
( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter
Granddaughters Lucy Romalotti Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on
the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed
the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like
Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 . Sheila is known as a villain . A significant
portion of Sheila 's history on both soaps revolves around her long - running conflict with Lauren Fenmore , whom she has attempted to kill on multiple
occasions .
</item 4>

<item 5>
Sheila Carter
She has also had conflicts with Stephanie Forrester , Maggie Forrester , Amber Moore , Taylor Hayes , Brooke Logan , Phyllis Summers and Quinn Fuller . Contents
( hide ) 1 Casting and creation 2 Development 3 Storylines 3.1 1990 -- 92 3.2 1992 -- 98 3.3 2002 -- 03 3.4 2005 -- 07 3.5 2010 , 2012 3.6 2017 -- 4 Critical
reception 5 References 6 External links Casting and creation ( edit ) The role was portrayed by Kimberlin Brown for a total of 16 years . Her first run was on
The Young and the Restless from May 1990 to May 20 , 1992 , when the character crossed over to The Bold and the Beautiful from May 21 , 1992 , to October 2 ,
1998 , again from May 24 to November 5 , 2002 , and again finally from September 26 to October 10 , 2003 .
</item 5>

<item 6>
Sheila Carter
Sheila continues to insinuate herself into Eric 's life , faking a concussion after a catfight with Quinn to manipulate Eric into letting her stay in the
Forrester mansion . Sheila blackmails her ex-husband , James Warwick , into lying about the severity of her condition . Critical reception ( edit ) Sheila is
often considered to be one of the most popular villains in daytime . Kimberlin Brown was nominated for a Daytime Emmy in the role as Outstanding Supporting
Actress in 1993 . With the global success of The Bold and the Beautiful , Sheila is arguably the most recognisable soap opera villain in the world . References
( edit ) ^ Jump up to : `` About Y&R : Who 's Who in Los Angeles - Sheila Carter '' . SoapCental . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. (
9 July 2005 ) . `` Y&R confirms the return of Kimberlin Brown '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday
, 5th August 2005 '' . CBS.com .
</item 6>

<item 7>
Sheila Carter
Sheila Carter - Wikipedia Sheila Carter Jump to : navigation , search Sheila Carter Kimberlin Brown as Sheila Carter The Young and the Restless character
Portrayed by Kimberlin Brown ( 1990 -- ) Michelle Stafford ( 2006 -- 07 ) Duration 1990 -- 98 2002 -- 03 2005 -- 07 2017 -- First appearance May 1990 Created by
William J. Bell Introduced by Edward J. Scott ( 1990 , 1992 -- 95 ) Lee Phillip Bell ( 1992 ) Bradley Bell ( 2002 -- 03 ) John F. Smith ( 2005 ) Lynn Marie
Latham and Josh Griffith ( 2006 ) Bradley Bell and Michael Minnis ( 2017 ) Crossover appearances The Bold and the Beautiful Classification Present ; regular
Profile Other names Sheila Grainger Sheila Forrester Sheila Warwick Sybil Weller Sheila Lovejoy Brenda Harris Jennifer Mitchell Phyllis Newman Betsy Occupation
Waitress Executive assistant Nurse Residence Los Angeles , California Michelle Stafford as Sheila Carter ( show ) Family Family Carter Mother Molly Carter
Sisters Sarah Smythe Husband Scott Grainger , Sr.
</item 7>

<item 8>
Sheila Carter
Jump up ^ Kimberlin Brown ( 2008 ) . Planet Soap - Sheila & Lauren on Catfights . TV Tropolis . Retrieved 16 September 2012 . ^ Jump up to : Kimberlin Brown (
2008 ) . Planet Soap - Soap Villains Part 1 . TV Tropolis . Retrieved 16 September 2012 . Jump up ^ `` Top 10 Soap Opera Villains of All Time '' .
soaps.about.com . Retrieved 25 August 2014 . Jump up ^ `` Soap opera slimeballs : The best villains '' . www.hlntv.com . Retrieved 25 August 2014 . External
links ( edit ) Sheila Carter biography at SoapCentral The Young and the Restless Cast Present cast members Past cast members Crew William J. Bell Lee Phillip
Bell Mal Young Families Newman Abbott Chancellor Barber / Winters Related articles Genoa City Storylines The Bold and the Beautiful Victor and Nikki Nick and
Sharon J.T.
</item 8>

<item 9>
Sheila Carter
Picking up a discarded newspaper in an area diner , she read about an opening working as a company nurse for the Forresters in Los Angeles . She left Wisconsin
to travel to California . Meanwhile , Lauren is reunited with Scott and her real son , Scott Jr . 1992 -- 98 ( edit ) In 1992 , Sheila is reintroduced on The
Bold and the Beautiful and she quickly worms her way into fashion CEO Eric Forrester ( John McCook ) 's life by becoming the company nurse at Forrester
Creations in Los Angeles , as well as working as Rick Forrester 's nanny ( after staging an accident that leaves the former nanny , Judy , hospitalized , and
later threatening Judy and her family ) . Sheila and Eric fall in love and are eventually engaged , which angers Eric 's ex-wife Stephanie Forrester ( Susan
Flannery ) , who suspects Sheila is a gold digger and a liar . To Sheila 's dismay , Eric 's most recent ex-wife Brooke Logan Forrester ( Katherine Kelly Lang )
is pregnant with a child that may either belong to Eric or his son Ridge Forrester ( Ronn Moss ) .
</item 9>

<item 10>
Sheila Carter
. Daytime Confidential . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. ( 30 November 2009 ) . `` Is Kimberlin Brown returning to The Young and the
Restless ? '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ Logan , Michael ( June 9 , 2017 ) . `` Kimberlin Brown Returns to The Bold and the
Beautiful as Supervillain Sheila Carter '' . TV Insider . United States : CBS Interactive . Retrieved June 10 , 2017 . Jump up ^ Logan , Michael ( June 19 ,
2017 ) . `` The Bold and the Beautiful : EP Bradley Bell Answers Our Burning Sheila Carter Questions '' . TV Insider . United States : TVGM Holdings LLC .
Retrieved June 19 , 2017 . Jump up ^ Kimberlin Brown ( 1992 ) . The Bold and the Beautiful May 1992 Promo Sheila . CBS Daytime . Retrieved 7 November 2012 .
Jump up ^ Kimberlin Brown ( 2008 ) .
</item 10>
=== END ITEM id=abcfc8a3d9bceb6b ===

=== ITEM id=b623614479b4325a ===
Gold sentences and facts:
Sentence 0: At the federal level in the United States , legislation ( i.e. , `` statutes '' or `` statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either signed into law by the President or passed by Congress after a presidential veto .
  Fact 0.0: At the federal level in the United States, legislation consists exclusively of Acts.
  Fact 0.1: At the federal level in the United States, legislation is also called statutes or statutory law.
  Fact 0.2: Federal legislation in the United States consists exclusively of Acts passed by the Congress of the United States.
  Fact 0.3: Federal legislation in the United States consists exclusively of Acts passed by the Continental Congress, the predecessor of the Congress of the United States.
  Fact 0.4: Acts passed by the Congress of the United States or the Continental Congress were signed into law by the President.
  Fact 0.5: Acts passed by the Congress of the United States or the Continental Congress were passed by Congress after a presidential veto.

Retrieved items:
<item 1>
State legislature (United States)
Upon introduction , a bill is usually read by its title only , constituting the first reading of the bill . Because a bill is usually read by title only , it is
important that the title give the members notice of the subject matter contained in the bill . Committees ( edit ) General information ( edit ) As with other
legislative bodies throughout the world , U.S. state legislatures operate mainly through committees when considering proposed bills . Thus , committee action is
probably the most important phase of the legislative process . Most bills can not be enacted into law until it has been referred to , acted upon by , and
returned from , a standing committee in each house . Reference to committee usually follows the first reading of the bill . Each committee is set up to consider
bills relating to a particular subject . Standing committees are charged with the important responsibility of examining bills and recommending action to the
Senate or House . Often on days when a legislature is not in session , the committees of each house meet and consider the bills that have been referred to them
to decide if the assigned bills should be reported for further action .
</item 1>

<item 2>
List of United States federal legislation
( edit ) Acts of Congress are published in the United States Statutes at Large . Volumes 1 through 18 , which have all the statutes passed from 1789 to 1875 ,
are available on - line at the Library of Congress , here . In the list below , statutes are listed by X Stat . Y , where X is the volume of the Statutes at
Large and Y is the page number , as well as either the chapter or Public Law number . See examples below . Sessions ( Sess. . ) and Chapters ( ch. . ) ( edit )
Each Congress has two to four sessions . Under the numbering system used from 1789 until 1957 , the Acts in each session are numbered sequentially as Chapters .
This numbering included both laws applicable to the general public and laws relating to specific individuals , e.g. , to grant pensions to disabled veterans .
Public laws ( edit ) Since 1957 , Acts of Congress related to the general public have been designated in the form : Public Law X-Y where X is the number of the
ordinal Congress and Y is the number of the chronological order of the public Act in that Congress .
</item 2>

<item 3>
List of United States federal legislation
This numbering is continuous across sessions of Congress and does not include acts relating to individuals , which are analogously given Private Law X-Y
designations . ( From 1901 to 1957 , the Statutes at Large listed both a chapter and a Public / Private number for each act ) See examples below. There are many
laws . Examples ( edit ) The Militia Act of 1862 of July 17 , 1862 , Sess. 2 , ch. 201 , was the 201st Act of the second session of the 37th Congress . The
National Banking Act of February 25 , 1863 , Sess. 3 , ch. 58 , was the 58th Act of the third session of the 37th Congress . The Global Anti-Semitism Review Act
of 2004 of October 16 , 2004 , Pub. L. 108 -- 332 , 118 Stat. 1282 , was the 332nd Act of Congress ( statute ) passed in the 108th Congress . It can be found in
volume 118 of the U.S. Statutes at Large , starting at page 1282 .
</item 3>

<item 4>
List of United States federal legislation
Template : US tax acts , and Category : United States federal taxation legislation Transportation : United States Department of Transportation # Related
legislation Veterans Affairs : United States Department of Veterans Affairs # Related legislation Water Resources , Navigation , Environmental Regulation :
United States Army Corps of Engineers # Public Laws affecting the Corps of Engineers ( partial ) Sources ( edit ) Acts listed by popular name , via Cornell
University Statutes at Large Volumes 1 through 18 , 1789 -- 1875 , via Library of Congress Volumes 19 through 64 , 1875 - 1950 , via Library of Congress Volumes
65 through 125 , 1951 - 2011 , via Government Publishing Office Public laws 93rd Congress through current Congress , via Congress.gov U.S. Code U.S. Code , via
Law Revision Counsel of the U.S. House of Representatives U.S. Code , via Cornell University U.S. Code , via the U.S. Government Publishing Office U.S. Code ,
via FindLaw.com Brian K. Landsberg ( ed ) , Major Acts of Congress .
</item 4>

<item 5>
State legislature (United States)
If the majority vote in favor of the bill , it is recorded as passed . Transmission to second house ( edit ) A bill that is passed in one house is transmitted ,
along with a formal message , to the other house . If the bill is not reported from committee or is not considered by the full house , the bill is defeated .
The house of origin , upon return of its amended bill , may take any one of several courses of action . It may concur in the amendment by the adoption of a
motion to that effect ; then the bill , having been passed by both houses in identical form , is ready for enrollment . Another possibility is that the house of
origin may adopt a motion to non-concur in the amendment , at which point the bill dies . Finally , the house of origin may refuse to accept the amendment but
request that a conference committee be appointed . The other house usually agrees to the request , and the presiding officer of each house appoints members to
the conference committee .
</item 5>

<item 6>
List of United States federal legislation
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power . See also : Executive orders of the President
; regulations of Executive branch departments and administrative agencies ; and the procedural rules of the federal courts . Contents ( hide ) 1 Publication of
the law 1.1 Statutes at Large ( Stat . ) 1.2 Sessions ( Sess . ) and Chapters ( ch . ) 1.3 Public laws 1.4 Examples 2 Congress of the Confederation 3 United
States Congress 3.1 1789 to 1901 : 1st through 56th Congresses 3.2 1901 to 2001 : 57th through 106th Congresses 3.3 2001 to present : 107th and subsequent
Congresses 4 See also 5 Sources 6 References Publication of the law ( edit ) Statutes at Large ( Stat. . ) ( edit ) Acts of Congress are published in the United
States Statutes at Large .
</item 6>

<item 7>
List of United States federal legislation
List of United States federal legislation - wikipedia List of United States federal legislation Jump to : navigation , search This article includes a list of
references , but its sources remain unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise
citations. ( April 2016 ) ( Learn how and when to remove this template message ) See also : Category : United States federal legislation This is a chronological
, but still incomplete , list of United States federal legislation . Congress has enacted approximately 200 -- 600 statutes during each of its 115 biennial
terms , so that more than 20,000 statutes have been enacted since 1789 . At the federal level in the United States , legislation ( i.e. , `` statutes '' or ``
statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either
signed into law by the President or passed by Congress after a presidential veto . Legislation is not the only source of regulations with the force of law .
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power .
</item 7>

<item 8>
State legislature (United States)
For most bills , the recommendations of the committee are followed , although either house is free to accept or reject the action of the committee . Bills
reported favorably by a committee may be placed on a regular calendar ( the agenda of the deliberative body ) . Most of the work of the legislature is done by
committees . The legislature as a whole relies on its committees to report out only those bills deserving the consideration of the entire house . Through
standing committees , each bill is addressed by a group of members who have special knowledge of its subject . Some members of the legislature have expert
knowledge of particular subjects of legislation , and these members are usually placed on committees to take full advantage of this specialized knowledge . For
this reason , the legislature often accepts the final recommendations of its standing committees . As has been noted , however , the legislature does not
completely abdicate its responsibility for the consideration of pending bills . If the need arises , the members of either house can force a committee to take
action on a bill , or they can ignore the committee 's recommendations .
</item 8>

<item 9>
State legislature (United States)
If the governor does not approve of the bill , he or she may veto it . In the event of a veto , the governor returns the bill to the house in which it
originated , with a message explaining his or her objections and suggesting any amendments ( if applicable ) which might remove those objections . The bill is
then reconsidered , and if a simple majority of the members of both houses agrees to the proposed executive amendments , it is returned to the Governor , as he
or she revised it , for his or her signature . On the other hand , a simple majority of the members of each house can choose to approve a vetoed bill precisely
as the Legislature originally passed it , in which case it becomes a law over the governor 's veto . This is in contrast to the practice in most states and the
federal government , which require a two - thirds majority in both houses to override a governor 's veto . If the Governor fails to return a bill to the
legislative house in which it originated within a specified number of days after it was presented to him or her , it becomes a law without their signature .
</item 9>

<item 10>
State legislature (United States)
Conference committees ( edit ) A conference committee is often empaneled to discuss the points of difference between the two houses ' versions of the same bill
, and tries to reach an agreement between them so that the identical bill can be passed by both houses . If an agreement is reached and if both houses adopt the
conference committee 's report , the bill is passed . If either house refuses to adopt the report of the conference committee , a motion may be made for further
conference . If a conference committee is unable to reach an agreement , it may be discharged , and a new conference committee may be appointed . Some highly
controversial bills may be referred to several different conference committees . If an agreement is never reached in conference prior to the end of the
legislative session , the bill is lost . When a bill has passed both houses in identical form , it is then ready for transmittal to the Governor . Presentation
to the governor ( edit ) Once a bill reaches the governor , he or she may sign it , which completes its enactment into law . From this point , the bill becomes
an act , and remains the law of the state unless repealed by legislative action , or overturned by a court decision .
</item 10>
=== END ITEM id=b623614479b4325a ===

=== ITEM id=f668ae356549d5ce ===
Gold sentences and facts:
Sentence 0: Originally an ordinary man by the name of Nick Chopper ( the name first appearing in The Marvelous Land of Oz ) , the Tin Woodman used to make his living chopping down trees in the forests of Oz , as his father had before him . The Wicked Witch of the East enchanted his axe to prevent him from marrying his sweetheart , after being bribed by the lazy old woman who kept the Munchkin maiden as a servant , and did not wish to lose her . ( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick 's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no longer able to love the lady he had fallen for .
  Fact 0.0: Nick Chopper was originally an ordinary man.
  Fact 0.1: The name Nick Chopper first appears in The Marvelous Land of Oz.
  Fact 0.2: Nick Chopper used to make his living chopping down trees in the forests of Oz.
  Fact 0.3: Nick Chopper's father had also chopped down trees in the forests of Oz.
  Fact 0.4: The Wicked Witch of the East enchanted Nick Chopper's axe to prevent him from marrying his sweetheart.
  Fact 0.5: The Wicked Witch of the East was bribed to do this by the lazy old woman who kept the Munchkin maiden as a servant.
  Fact 0.6: The lazy old woman did not wish to lose the Munchkin maiden servant.
  Fact 0.7: In a later book of the series, The Tin Woodman of Oz, the woman is said to be the Witch's servant.
  Fact 0.8: In The Tin Woodman of Oz, the Witch herself decides to enchant Nick's axe.
  Fact 0.9: The enchanted axe chopped off Nick Chopper's limbs one by one.
  Fact 0.10: Each time Nick lost a limb, Ku-Klip the tinsmith replaced it with a prosthetic limb made of tin.
  Fact 0.11: Finally, nothing was left of Nick Chopper but tin.
  Fact 0.12: Ku-Klip neglected to replace Nick Chopper's heart.
  Fact 0.13: Once Nick Chopper was made entirely of tin, he was no longer able to love the lady he had fallen for.
Sentence 1: In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest , as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West . The Tin Woodman 's axe proves useful in this journey , both for chopping wood to create a bridge or raft as needed , and for chopping the heads off animals that threaten the party . When the Winged monkeys are sent by the Witch of the West against the group , they throw the Tin Woodman from a great height , damaging him badly . However Winkie Tinsmiths are able to repair him after the death of the Witch .
  Fact 1.0: In The Wonderful Wizard of Oz, Dorothy Gale befriends the Tin Woodman.
  Fact 1.1: Dorothy and her friends find the Tin Woodman rusted in the forest.
  Fact 1.2: The Tin Woodman was caught in rain.
  Fact 1.3: Dorothy uses the Tin Woodman's oil can to release him.
  Fact 1.4: The Tin Woodman follows Dorothy to the Emerald City to get a heart from the Wizard.
  Fact 1.5: The Scarecrow and the Cowardly Lion join Dorothy and the Tin Woodman on their adventure.
  Fact 1.6: The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West.
  Fact 1.7: The Tin Woodman's axe proves useful on this journey for chopping wood to create a bridge or raft as needed.
  Fact 1.8: The Tin Woodman's axe is useful for chopping the heads off animals that threaten the party.
  Fact 1.9: The Winged Monkeys are sent by the Witch of the West against the group.
  Fact 1.10: The Winged Monkeys throw the Tin Woodman from a great height.
  Fact 1.11: The fall damages the Tin Woodman badly.
  Fact 1.12: Winkie Tinsmiths are able to repair the Tin Woodman after the death of the Witch.
Sentence 2: His desire for a heart notably contrasts with the Scarecrow 's desire for brains , reflecting a common debate between the relative importance of the mind and the emotions . This occasions philosophical debate between the two friends as to why their own choices are superior ; neither convinces the other , and Dorothy , listening , is unable to decide which one is right . Symbolically , because they remain with Dorothy throughout her quest , she is provided with both and need not select . The Tin Woodman states unequivocally that he has neither heart nor brain , but cares nothing for the loss of his brain . Towards the end of the novel , though , Glinda praises his brain as not quite that of the Scarecrow 's .
  Fact 2.0: The Tin Woodman's desire for a heart notably contrasts with the Scarecrow's desire for brains.
  Fact 2.1: The contrast reflects a common debate between the relative importance of the mind and the emotions.
  Fact 2.2: The two friends have a philosophical debate about why their own choices are superior.
  Fact 2.3: Neither friend convinces the other in that debate.
  Fact 2.4: Dorothy, listening, is unable to decide which one is right.
  Fact 2.5: Symbolically, because the Tin Woodman and the Scarecrow remain with Dorothy throughout her quest, she is provided with both a heart and a brain and need not select.
  Fact 2.6: The Tin Woodman states unequivocally that he has neither heart nor brain.
  Fact 2.7: The Tin Woodman cares nothing for the loss of his brain.
  Fact 2.8: Towards the end of the novel, Glinda praises the Tin Woodman's brain.
  Fact 2.9: Glinda says the Tin Woodman's brain is not quite that of the Scarecrow.

Retrieved items:
<item 1>
Tin Woodman
The movie shows him moving through a colorful Oz city with his brand - new mechanical heart , before meeting a reddish female Tin Woman ( or robot ? ) to whom
he gives the heart . She proceeds to cruelly play with the heart . An internet - collaboration , CG animated feature based on Baum 's book The Tin Woodman of Oz
was produced by A : M Films , and completed in 2009 . In 2010 , Whitestone Motion Pictures produced a 23 - minute live - action short film , Heartless : The
Story of the Tin Man . The film is based on the book The Tin Woodman of Oz , where Woot the Wanderer visits the Tin Man and asks how he came to be made of tin .
In the film , the Tin Man appears to be more steam - or coal - powered . His chest cavity is covered by a door which reads `` Pedudoe Tin Co . '' but this is a
reference to the workings of the film company , and not a reference to any Oz book or material . The movie was made available for free viewing online and free
downloading of its soundtrack .
</item 1>

<item 2>
Tin Woodman
Toward the end , he hugged the Wizard ( Joel Grey ) saying to him `` Thank you from the bottom of my heart ! '' . In 1996 , they made a cartoon animated The Oz
Kids . He has a son named Tin Boy . Like his father he can cut wood with his ax . And he has to be careful with water and snow or he 'll be rust . His voice
played by Andy Milder . Hiroki Tsujiai from Ultra Maniac dresses as the Tin Man in the costume party . James Kall portrayed the Tin Man in the 2001 ABC
miniseries Life with Judy Garland : Me and My Shadows . The Muppet Gonzo plays a similar role , the Tin Thing , in 2005 's television film The Muppets ' Wizard
of Oz . In this version , he is the Wicked Witch 's research assistant , transformed into a robot to prevent him wanting a day off to marry Camilla . Gonzo 's
other role is himself . He appears at the end of this film in the Muppets ' show .
</item 2>

<item 3>
Tin Woodman
In the musical adaptation of Wicked the Tin Man is revealed to be Boq , a Munchkin whom Nessarose , the Wicked Witch of the East , fell in love with when they
were at school together . When she discovered his heart belonged to Glinda , she botched a spell that was meant to make him fall in love with her by taking his
heart , but instead shrunk his heart to nothing by taking it away without ' giving ' it to Nessa . To save his life , Elphaba , the Wicked Witch of the West ,
was forced to turn him into tin . Not understanding her reasons , he pursues Elphaba with a single - minded vengeance for his current form . The Tin Man 's
humble origin in the novel conflicts with his having been the aristocratic Boq . In Oz Squad , Nick was shown in a sexual relationship with `` Rebecca Eastwitch
'' in order to get closer to Nimmie Amee and attempt to elope with her .
</item 3>

<item 4>
Tin Woodman
) The Magical Mimics in Oz ( 1946 ) The Shaggy Man of Oz ( 1949 ) The Hidden Valley of Oz ( 1951 ) Merry Go Round in Oz ( 1963 ) Yankee in Oz ( 1972 ) The
Enchanted Island of Oz ( 1976 ) The Forbidden Fountain of Oz ( 1980 ) The Ozmapolitan of Oz ( 1986 ) The Wicked Witch of Oz ( 1993 ) The Giant Garden of Oz (
1993 ) The Runaway in Oz ( 1995 ) The Rundelstone of Oz ( 2000 ) The Emerald Wand of Oz ( 2005 ) Trouble Under Oz ( 2006 ) Alternate Dorothy of Oz ( 1989 )
Locations and elements Land of Oz Emerald City Munchkin Country Gillikin Country Winkie Country Quadling Country Yellow brick road Oogaboo Deadly Desert
Nonestica Forest of Burzee Land of Ev Merryland Mo Ix Nome Kingdom Noland Ruby slippers Silver Shoes The Oz Film Manufacturing Company Characters Dorothy Gale
Toto Princess Ozma The Wizard of Oz Scarecrow Tin Woodman Cowardly Lion Jack Pumpkinhead Tik - Tok Sawhorse Trot Hungry Tiger
</item 4>

<item 5>
Tin Woodman
( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick
's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of
tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no
longer able to love the lady he had fallen for . In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest
, as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their
adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West .
</item 5>

<item 6>
Tin Woodman
The Tin Woodman 's axe proves useful in this journey , both for chopping wood to create a bridge or raft as needed , and for chopping the heads off animals that
threaten the party . When the Winged monkeys are sent by the Witch of the West against the group , they throw the Tin Woodman from a great height , damaging him
badly . However Winkie Tinsmiths are able to repair him after the death of the Witch . His desire for a heart notably contrasts with the Scarecrow 's desire for
brains , reflecting a common debate between the relative importance of the mind and the emotions . This occasions philosophical debate between the two friends
as to why their own choices are superior ; neither convinces the other , and Dorothy , listening , is unable to decide which one is right . Symbolically ,
because they remain with Dorothy throughout her quest , she is provided with both and need not select . The Tin Woodman states unequivocally that he has neither
heart nor brain , but cares nothing for the loss of his brain . Towards the end of the novel , though , Glinda praises his brain as not quite that of the
Scarecrow 's .
</item 6>

<item 7>
Tin Woodman
They built a story around them , added Dorothy , and added a series of lessons to the effect that everyone possesses the resources they need ( such as brains ,
a heart and courage ) if only they had self - confidence . The Wonderful Wizard of Oz was a children 's book , of course , but as Baum warned in the preface ,
it was a `` modernized '' fairy tale as well . The Tin Man -- the human turned into a machine -- was a common feature in political cartoons and in
advertisements in the 1890s . Indeed , he had been part of European folk art for 300 years . In political interpretations of The Wonderful Wizard of Oz , the
Tin Woodman is supposedly described as a worker , dehumanized by industrialization . The Tin Woodman little by little lost his natural body and had it replaced
by metal ; so he has lost his heart and can not move without the help of farmers ( represented by the Scarecrow ) ; in reality he has a strong sense of
cooperation and love , which needs only an infusion of self - confidence to be awakened . In the 1890s many argued that to secure a political revolution a
coalition of Farmers and Workers was needed .
</item 7>

<item 8>
Tin Woodman
To prevent himself from completely becoming tin , Stanum seeks out help from Zelena ( now the Wicked Witch of the West ) at the Emerald City of Oz . Zelena
agrees to help him seek out the Crimson Heart , which can save him . During their quest , Stanum tells Zelena that she does n't have to be lonely but she is
doing her best to deny his advice . Suddenly , a lion comes of nowhere to attack Stanum , and Zelena uses her magic to make the lion go away ( at this point the
lion has become as Zelena would put it , cowardly ) . When they finally arrived to the location of the Crimson Heart , the two learned that the only way to make
it work is through the absorption of another person 's magic . Unfortunately Zelena 's actions and selfish greed for magic causes her to betray Stanum , whom
she suspect was aligned with Dorothy by keeping the Crimson Heart for herself , leaving Stanum to transform into the Tin Man permanently . Some time later ,
when Robin Hood arrives in Oz in the episode `` Heart of Gold '' , the Tin Man is seen on the side of the Yellow Brick Road , torn apart .
</item 8>

<item 9>
Tin Woodman
The Wizard turns out to be a `` humbug '' and can only provide a placebo heart made of silk and filled with sawdust . However , this is enough to please the Tin
Woodman , who , with or without a heart , was all along the most tender and emotional of Dorothy 's companions ( just as the Scarecrow was the wisest and the
Cowardly Lion the bravest ) . When he accidentally crushes an insect , he is grief - stricken and , ironically , claims that he must be careful about such
things , while those with hearts do not need such care . This tenderness remains with him throughout the series , as in The Patchwork Girl of Oz , where he
refuses to let a butterfly be maimed for the casting of a spell . When Dorothy returns home to her farm in Kansas , the Tin Woodman returns to the Winkie
Country to rule as emperor . Later , he has his subjects construct a palace made entirely of tin -- from the architecture all the way down to the flowers in the
garden . Baum emphasized that the Tin Woodman remains alive , in contrast to the windup mechanical man Tik - Tok Dorothy meets in a later book .
</item 9>

<item 10>
Tin Woodman
When they get to the heart , Zelena picks it up and felt some of her magic draining . When Zelena claims that Dorothy Gale put Stanum up to this task , Stanum
commented that he never met Dorothy . He also stated that Zelena once claimed on the day they met that she would be happy being a normal human . As Stanum fully
turns to tin and is unable to move , Zelena takes the box containing the Crystal Heart and leaves while telling Stanum that perhaps Dorothy will meet Stanum and
free him . Modern works ( edit ) This article gives self - sourcing examples without describing their significance in the context of the article . Please help
improve this article by adding citations to reliable sources that describe the examples ' significance , and by removing less pertinent examples . Unsourced or
poorly sourced material may be challenged or removed . ( November 2017 ) ( Learn how and when to remove this template message ) The song `` Country Robot / A
Letter to Dorothy '' by The Incredible Moses Leroy is written from the Tin Man 's perspective ; it includes the lyrics `` You gave me oil , I was a rusty load /
You even helped me find my heart . ''
</item 10>
=== END ITEM id=f668ae356549d5ce ===

=== ITEM id=e48323cfae5b20cb ===
Gold sentences and facts:
Sentence 0: The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert .
  Fact 0.0: The Second Great Awakening occurred in several episodes.
  Fact 0.1: The Second Great Awakening occurred over different denominations.
  Fact 0.2: The revivals of the Second Great Awakening were very similar to each other.
  Fact 0.3: Revival meetings were the most effective form of evangelizing during this period.
  Fact 0.4: Revival meetings cut across geographical boundaries.
  Fact 0.5: The revival movement quickly spread throughout Kentucky, Tennessee and southern Ohio.
  Fact 0.6: Each denomination had assets that allowed it to thrive on the frontier.
  Fact 0.7: The Methodists had an efficient organization.
  Fact 0.8: The Methodist organization depended on itinerant ministers known as circuit riders.
  Fact 0.9: Circuit riders sought out people in remote frontier locations.
  Fact 0.10: Circuit riders came from among the common people.
  Fact 0.11: Coming from among the common people helped circuit riders establish rapport with the frontier families they hoped to convert.
Sentence 1: The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ .
  Fact 1.0: The revivals enrolled millions of new members in existing evangelical denominations.
  Fact 1.1: The revivals led to the formation of new denominations.
  Fact 1.2: Many converts believed that the Awakening heralded a new millennial age.
  Fact 1.3: The Second Great Awakening stimulated the establishment of many reform movements.
  Fact 1.4: The reform movements were designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ.
Sentence 2: In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be saved was from grace from God .
  Fact 2.0: In the newly settled frontier regions, the revival was implemented through camp meetings.
  Fact 2.1: Camp meetings often provided the first encounter for some settlers with organized religion.
  Fact 2.2: Camp meetings were important as social venues.
  Fact 2.3: A camp meeting was a religious service of several days' length with preachers.
  Fact 2.4: Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship.
  Fact 2.5: Participating in a religious revival with crowds of hundreds and perhaps thousands of people was sheer exhilaration.
  Fact 2.6: That exhilaration inspired the dancing, shouting, and singing associated with camp meeting events.
  Fact 2.7: The revivals followed an arc of great emotional power.
  Fact 2.8: The revivals emphasized the individual's sins and need to turn to Christ.
  Fact 2.9: The revivals were restored by a sense of personal salvation.
  Fact 2.10: Upon their return home, most converts joined or created small local churches.
  Fact 2.11: The small local churches grew rapidly.
  Fact 2.12: The Second Great Awakening marked a religious transition in society in America.
  Fact 2.13: Many Americans from the Calvinist sect emphasized man's inability to save themselves.
  Fact 2.14: Many Americans from the Calvinist sect emphasized that their only way to be saved was by grace from God.

Retrieved items:
<item 1>
Second Great Awakening
Contents ( hide ) 1 Spread of revivals 1.1 Background 1.2 Theology 1.3 Burned - over district 1.4 West and Tidewater South 1.5 West 1.6 Church membership soars
2 Subgroups 2.1 Adventism 2.2 Holiness movement 2.3 Restoration Movement 3 Culture and society 4 Slaves and free Africans 5 Women 6 Prominent figures 7
Political implications 8 See also 9 References 10 Further reading 10.1 Historiography Spread of revivals ( edit ) Part of a series on Baptists Background ( show
) Christianity Protestantism Puritanism Anabaptism Doctrine ( show ) Priesthood of all believers Individual soul liberty Separation of church and state Sola
scriptura Congregationalism Ordinances Offices Confessions Key figures ( show ) John Smyth Thomas Helwys Roger Williams John Clarke John Bunyan Shubal Stearns
Andrew Fuller Charles Spurgeon D.N. Jackson James Robinson Graves William Bullein Johnson William Carey Luther Rice Martin Luther King Jr .
</item 1>

<item 2>
Second Great Awakening
While Protestant religion had previously played an important role on the American political scene , the Second Great Awakening strengthened the role it would
play . See also ( edit ) Baptist portal Methodism portal Calvinism portal Seventh - day Adventist Church portal Latter Day Saints portal LDS Church portal
Advent Christian Church Christian revival Christianity in the 19th century The Church of Jesus Christ of Latter - day Saints Cumberland Presbyterian Church
Ethnocultural politics in the United States Holiness movement Restoration Movement Seventh - day Adventist Church References ( edit ) ^ Jump up to : Timothy L.
Smith , Revivalism and Social Reform : American Protestantism on the Eve of the Civil War ( 1957 ) . Jump up ^ Heyrman , Christine Leigh . `` The First Great
Awakening '' . Divining America , TeacherServe . National Humanities Center . Jump up ^ Henry B. Clark ( 1982 ) . Freedom of Religion in America : Historical
Roots , Philosophical Concepts , Contemporary Problems . Transaction Publishers . p. 16 . ^ Jump up to : Nancy Cott , `` Young Women in the Second Great
Awakening in New England , '' Feminist Studies , ( 1975 ) , 3 # 1 p.
</item 2>

<item 3>
Second Great Awakening
The Second Great Awakening served as an `` organizing process '' that created `` a religious and educational infrastructure '' across the western frontier that
encompassed social networks , a religious journalism that provided mass communication , and church - related colleges . Publication and education societies
promoted Christian education ; most notable among them was the American Bible Society , founded in 1816 . Women made up a large part of these voluntary
societies . The Female Missionary Society and the Maternal Association , both active in Utica , NY , were highly organized and financially sophisticated women
's organizations responsible for many of the evangelical converts of the New York frontier . There were also societies that broadened their focus from
traditional religious concerns to larger societal ones . These organizations were primarily sponsored by affluent women . They did not stem entirely from the
Second Great Awakening , but the revivalist doctrine and the expectation that one 's conversion would lead to personal action accelerated the role of women 's
social benevolence work . Social activism influenced abolition groups and supporters of the Temperance movement . They began efforts to reform prisons and care
for the handicapped and mentally ill . They believed in the perfectibility of people and were highly moralistic in their endeavors .
</item 3>

<item 4>
Second Great Awakening
The influence of the Awakening continued in the form of more secular movements . In the midst of shifts in theology and church polity , American Christians
began progressive movements to reform society during this period . Known commonly as antebellum reform , this phenomenon included reforms in against the
consumption of alcohol , for women 's rights and abolition of slavery , and a multitude of other issues faced by society . The religious enthusiasm of the
Second Great Awakening was echoed by the new political enthusiasm of the Second Party System . More active participation in politics by more segments of the
population brought religious and moral issues into the political sphere . The spirit of evangelical humanitarian reforms was carried on in the antebellum Whig
party . Historians stress the understanding common among participants of reform as being a part of God 's plan . As a result , local churches saw their roles in
society in purifying the world through the individuals to whom they could bring salvation , and through changes in the law and the creation of institutions .
Interest in transforming the world was applied to mainstream political action , as temperance activists , antislavery advocates , and proponents of other
variations of reform sought to implement their beliefs into national politics .
</item 4>

<item 5>
Second Great Awakening
the Second reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skepticism , deism , and
rationalism left over from the Enlightenment . At about the same time , similar movements flourished in Europe . Pietism was sweeping German countries .
Evangelicalism was waxing strong in England . The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals
were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement
quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had
an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit
riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert . Theology ( edit ) Main article
: Postmillennialism Postmillennialism theology dominated American Protestantism in the first half of the 19th century .
</item 5>

<item 6>
Second Great Awakening
Second Great Awakening - wikipedia Second Great Awakening Jump to : navigation , search An engraving of a Methodist camp meeting in 1819 ( Library of Congress )
. Great Awakening First ( c. 1730 -- 1755 ) Second ( c. 1790 -- 1840 ) Third ( c. 1855 -- 1900 ) Fourth ( c. 1960 -- 1980 ) The Second Great Awakening was a
Protestant religious revival during the early 19th century in the United States . The movement began around 1790 , gained momentum by 1800 and , after 1820 ,
membership rose rapidly among Baptist and Methodist congregations whose preachers led the movement . It was past its peak by the late 1850s . The Second Great
Awakening reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skeptical rationalism and deism of
the Enlightenment . The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many
converts believed that the Awakening heralded a new millennial age .
</item 6>

<item 7>
Second Great Awakening
a re-emphasis on Wesleyan teachings on sanctification emerged during the Second Great Awakening , leading to a distinction between Mainline Methodism and
Holiness churches . Restoration movement ( edit ) Main article : Restoration Movement See also : Restorationism The idea of restoring a `` primitive '' form of
Christianity grew in popularity in the U.S. after the American Revolution . This desire to restore a purer form of Christianity without an elaborate hierarchy
contributed to the development of many groups during the Second Great Awakening , including the Mormons , Baptists and Shakers . Several factors made the
restoration sentiment particularly appealing during this time period : To immigrants in the early 19th century , the land in the United States seemed pristine ,
edenic and undefiled -- `` the perfect place to recover pure , uncorrupted and original Christianity '' -- and the tradition - bound European churches seemed
out of place in this new setting . A primitive faith based on the Bible alone promised a way to sidestep the competing claims of the many denominations
available and for congregations to find assurance of being right without the security of an established national church . The Restoration Movement began during
, and was greatly influenced by , the Second Great Awakening .
</item 7>

<item 8>
Second Great Awakening
The Restoration Movement began during , and was greatly influenced by , the Second Great Awakening . While the leaders of one of the two primary groups making
up this movement , Thomas Campbell and Alexander Campbell , resisted what they saw as the spiritual manipulation of the camp meetings , the revivals contributed
to the development of the other major branch , led by Barton W. Stone . The Southern phase of the Awakening `` was an important matrix of Barton Stone 's reform
movement '' and shaped the evangelistic techniques used by both Stone and the Campbells . Culture and society ( edit ) Efforts to apply Christian teaching to
the resolution of social problems presaged the Social Gospel of the late 19th century . Converts were taught that to achieve salvation they needed not just to
repent personal sin but also work for the moral perfection of society , which meant eradicating sin in all its forms . Thus , evangelical converts were leading
figures in a variety of 19th century reform movements . Congregationalists set up missionary societies to evangelize the western territory of the northern tier
. Members of these groups acted as apostles for the faith , and also as educators and exponents of northeastern urban culture .
</item 8>

<item 9>
Second Great Awakening
West ( edit ) Main article : Revival of 1800 In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided
the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several
days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of
participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these
events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of
personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a
religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be
saved was from grace from God . The Revival of 1800 in Logan County , Kentucky , began as a traditional Presbyterian sacramental occasion .
</item 9>

<item 10>
Second Great Awakening
Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements
designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ . Historians named the Second Great Awakening in the context of the
First Great Awakening of the 1730s and 1740s and of the Third Great Awakening of the late 1850s to early 1900s . These revivals were part of a much larger
Romantic religious movement that was sweeping across Europe at the time , mainly throughout England , Scotland , and Germany .
</item 10>
=== END ITEM id=e48323cfae5b20cb ===

=== ITEM id=e7f69c05b5889ecc ===
Gold sentences and facts:
Sentence 0: There are several major additional characters introduced by the television series and numerous narrative details and the plotline differ radically from the source novel . For example , the planned Nazi pre-emptive nuclear strike on Japan , `` Operation Dandelion , '' is apparently being prevented only by Hitler 's personal refusal to authorise it , leading Heydrich and the faction demanding pre-emptive war to plot the Führer 's assassination . In addition , Hawthorne Abendsen does not appear in the first season of the television version and The Grasshopper Lies Heavy is a series of newsreel films depicting multiple alternative realities rather than a novel ( although this idea may actually be borrowed from Dick 's later novel Valis which features a mysterious film depicting yet another dystopian alternative history of the USA ) . As of the Season 1 finale , these films are being tracked down by SS agents like Blake for dispatch to Hitler for an as - yet - unknown purpose .
  Fact 0.0: The television series introduces several major additional characters.
  Fact 0.1: The narrative details of the television series differ radically from the source novel.
  Fact 0.2: The plotline of the television series differs radically from the source novel.
  Fact 0.3: In the television series, the planned Nazi pre-emptive nuclear strike on Japan, 'Operation Dandelion', is apparently being prevented only by Hitler's personal refusal to authorise it.
  Fact 0.4: Hitler's refusal to authorise Operation Dandelion leads Heydrich and the faction demanding pre-emptive war to plot the Führer's assassination.
  Fact 0.5: Hawthorne Abendsen does not appear in the first season of the television version.
  Fact 0.6: In the television version, The Grasshopper Lies Heavy is a series of newsreel films depicting multiple alternative realities.
  Fact 0.7: In the television version, The Grasshopper Lies Heavy is not a novel.
  Fact 0.8: The idea of The Grasshopper Lies Heavy as newsreel films may actually be borrowed from Dick's later novel Valis.
  Fact 0.9: The novel Valis features a mysterious film depicting yet another dystopian alternative history of the USA.
  Fact 0.10: As of the Season 1 finale, these films are being tracked down by SS agents like Blake.
  Fact 0.11: The films are being tracked down for dispatch to Hitler.
  Fact 0.12: The films are being dispatched to Hitler for an as-yet-unknown purpose.

Retrieved items:
<item 1>
The Man in the High Castle
Contents ( hide ) 1 Synopsis 1.1 Background 1.2 Plot summary 2 The Grasshopper Lies Heavy 3 Inspirations 4 Reception 5 Adaptations 5.1 Audiobook 5.2 Television
6 Incomplete sequel 7 See also 8 References 9 Further reading 10 External links Synopsis ( edit ) Fictional map of the world after the Axis victory Briefly ,
The Man in the High Castle is a `` fictional picture of a world divided by Germany and Japan , victors of the second World War '' . Background ( edit ) In the
novel 's alternate history , U.S. President Franklin D. Roosevelt was assassinated by Giuseppe Zangara in 1934 leading to the continuation of the Great
Depression and U.S. isolationism during the opening of World War II . Adolf Hitler led Nazi Germany to conquer most of Europe and the Soviet Union , murdering
Jews , Roma , Slavs , and other groups . Meanwhile , Imperial Japan occupied China , before taking control of India and Oceania . The Nazis then helped Italy
conquer most of Africa . As Japan invaded the U.S.
</item 1>

<item 2>
The Man in the High Castle
Childan 's store is stocked in part with antiques from the Wyndam - Matson Corporation , a metalworking company . Frank Frink ( formerly Fink ) , a secretly
Jewish -- American veteran of World War II , has just been fired from the Wyndam - Matson factory , when he agrees to join a former coworker to begin a
handcrafted jewelry business . Meanwhile , Frink 's ex-wife , Juliana , works as a judo instructor in Cañon City , Colorado ( in the neutral buffer zone of
Mountain States ) , where she begins a sexual relationship with an Italian truck driver and ex-soldier , Joe Cinnadella . Throughout the book , many of these
characters frequently make important decisions using prophetic messages they interpret from the I Ching . Many characters are also reading a widely banned yet
extremely popular new novel , The Grasshopper Lies Heavy , which depicts an alternate history in which the Allies won World War II , a concept that amazes and
intrigues its readers .
</item 2>

<item 3>
The Man in the High Castle
In the novel , the Italian Empire is a minor power that controls North Africa and the Middle East ; in the series , however , it is shown through maps that
these territories are part of the Nazi Empire , suggesting that either the Italian Empire was annexed after the war or is self - governing within the Reich .
Characters from the book that do appear are in most cases far more fleshed out with deeper and sometimes rather different backstories than their novel originals
. For instance , Wegener is a Standartenführer in the SS , rather than a naval captain ( and oddly there are no German military or naval -- as opposed to SS --
personnel depicted anywhere in the first season ) . Rather than being a member of an organized internal resistance ( and despite his relatively low rank )
Wegener is a close personal confidante of Hitler and his disillusionment with the regime appears to be largely personal . Juliana and Frank are unmarried , but
living together , rather than divorced and separated . Frank has a sister , nephew and niece , although they are killed early in the series , and this propels
him into a more active role in relation to the resistance .
</item 3>

<item 4>
The Man in the High Castle
Yakima Herald . Retrieved March 28 , 2017 . Jump up ^ Andreeva , Nellie ( July 24 , 2014 ) . `` Amazon Studios Adds Drama ' The Man In The High Castle ' ,
Comedy ' Just Add Magic ' To Pilot Slate '' . Deadline . Retrieved January 10 , 2016 . Jump up ^ `` The Man in the High Castle : Season 1 , Episode 1 '' .
Amazon.com . Retrieved January 17 , 2015 . Jump up ^ `` The Man in the High Castle '' . Internet Movie Database . Retrieved January 18 , 2015 . Jump up ^ Lewis
, Hilary ( February 18 , 2015 ) . `` Amazon Orders 5 New Series Including ' Man in the High Castle ' '' . The Hollywood Reporter . Retrieved December 10 , 2015
. Jump up ^ Robertson , Adi ( February 18 , 2015 ) . `` Amazon green - lights The Man in the High Castle TV series '' . The Verge . Retrieved December 10 , 2015
.
</item 4>

<item 5>
The Man in the High Castle
The Nazis have drained the Mediterranean to make room for farmland , developed and used the hydrogen bomb , and designed rockets for extremely fast travel
across the world as well as space , having colonized the Moon , Venus , and Mars . The novel is set mostly in San Francisco in the P.S.A. ; here , Chinese
residents first appear in the novel as second - class citizens and black people as slaves . The secondary setting of the novel is the Rocky Mountains States ,
namely the cities of Cañon City , Denver and Cheyenne . Plot summary ( edit ) In 1962 , fifteen years after Imperial Japan and Nazi Germany have won World War
II , Robert `` Bob '' Childan owns an Americana antique shop in San Francisco , California ( located in the Japanese - occupied Pacific States of America ) ,
which is most commonly frequented by the Japanese , who make a fetish of romanticized American cultural artifacts . Childan is contacted by Nobusuke Tagomi , a
high - ranking Japanese trade official , who is seeking a gift to impress a visiting Swedish industrialist named Baynes .
</item 5>

<item 6>
The Man in the High Castle
Wegener flies back to Germany where upon landing learns that Reinhard Heydrich ( a member of the anti-Dandelion faction ) has launched a coup against Goebbels ,
possibly installing himself as Chancellor . Meanwhile , Tagomi remains shaken by the shootout and goes to Childan to sell back the gun he used in the fight ;
however , instead , sensing the energy from one of Frink 's jewelery items , Tagomi impulsively buys it from Childan , before undergoing a spiritually intense
ambiguous moment where he momentarily perceives an alternative - history version of San Francisco . Later , Tagomi on a whim forces the German authorities to
release Frink , whom Tagomi has never met and does not know is the maker of the jewelery . Juliana soon has her own spiritual experience when she arrives in
Cheyenne . There , she discovers that Abendsen now lives in a normal house with his family , having left behind the High Castle due to a change of outlook ; he
no longer preoccupies himself with thoughts that he might soon be assassinated .
</item 6>

<item 7>
The Man in the High Castle
The Man in the High Castle won the Hugo Award for Best Novel in 1963 . Beginning in 2015 , the book was adapted as a multi-season TV series , with Dick 's
daughter , Isa Dick Hackett , serving as one of the show 's producers . Reported inspirations include Ward Moore 's alternate Civil War history , Bring the
Jubilee ( 1953 ) , various classic World War II histories , and the I Ching ( referred to in the novel ) . The novel features a `` novel within the novel ''
comprising an alternate history within this alternate history wherein the Allies defeat the Axis ( though in a manner distinct from the actual historical
outcome ) .
</item 7>

<item 8>
The Man in the High Castle
In addition , Hawthorne Abendsen does not appear in the first season of the television version and The Grasshopper Lies Heavy is a series of newsreel films
depicting multiple alternative realities rather than a novel ( although this idea may actually be borrowed from Dick 's later novel Valis which features a
mysterious film depicting yet another dystopian alternative history of the USA ) . As of the Season 1 finale , these films are being tracked down by SS agents
like Blake for dispatch to Hitler for an as - yet - unknown purpose . In season two we see a map on John Smith 's wall of the world . From this map we can see
that Japan controls the entirety of the Pacific Ocean and most of the Asian continent , including China , India and half of Russia . Japan is also shown to
control Australia , New Zealand , western Canada , Alaska and , all pacific states in the USA . We also see that that the Japanese Empire controls the Yucatán
Peninsula and South American countries like Chile . In contrast , Nazi Germany is shown as controlling all of Europe , Africa and the Middle East .
</item 8>

<item 9>
The Man in the High Castle
A new paperback edition of the novel was published in 1992 by Vintage Books . Adaptations ( edit ) Audiobook ( edit ) An unabridged The Man in the High Castle
audiobook , read by George Guidall and running approximately 9.5 hours over seven audio cassettes , was released in 1997 . Another unabridged audiobook version
was released in 2008 by Blackstone Audio , read by Tom Wyner and running approximately 8.5 hours over seven CDs . A third unabridged audiobook recording was
released in 2014 by Brilliance Audio , read by Jeff Cummings with a running time of 9 hours 58 minutes . Television ( edit ) Main article : The Man in the High
Castle ( TV series ) After a number of attempts to adapt the book to the screen , in October 2014 Amazon 's film production unit began filming the pilot episode
of The Man in the High Castle in Roslyn , Washington , for release through the Amazon Prime Web video streaming service .
</item 9>

<item 10>
The Man in the High Castle
The pilot episode was released by Amazon Studios on January 15 , 2015 , and was Amazon 's `` most watched pilot ever '' according to Amazon Studios ' vice
president , Roy Price . On February 18 , 2015 , Amazon green - lit the series . The show became available for streaming on November 20 , 2015 . The television
series diverges from the novel in many significant respects . Both the Pacific States of America and the Eastern American puppet state appear to be mere
provinces of the Japanese and German empires without any apparent autonomous ( even quisling ) government institutions whatsoever . The Rocky Mountain States
become an anarchic Neutral Zone . World War II appears to have ended following the destruction of Washington , D.C. with an atomic bomb , rather than a land
invasion as in the book . As for Hitler , while elderly , he is apparently mostly hale in his Season 1 finale appearance , though other characters elsewhere in
the season do reference his supposed physical infirmity .
</item 10>
=== END ITEM id=e7f69c05b5889ecc ===
