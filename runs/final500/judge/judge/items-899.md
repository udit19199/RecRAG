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

=== ITEM id=c1f8f631c96b6d4d ===
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
Sheila Carter returned in December, portrayed by Michelle Stafford instead of Kimberlin Brown, who was recovering from an accident and unable to reprise the
role. Stafford played Sheila for two months, until Lauren Fenmore shot her on February 9, 2007, leaving her presumed dead. In 2009, rumors suggested Brown might
return to either *The Bold and the Beautiful* or *The Young and the Restless*; Brown confirmed Brad Bell had asked if she was interested, and she
</item 1>

<item 2>
Despite being a regular on *The Bold and the Beautiful*, Sheila Carter portrayer Brown made numerous guest appearances on *The Young and the Restless* from
October 1992 through March 1995. After a 10-year absence, she returned in August 2005, leaving *One Life to Live* after CBS offered better contract terms. Her
run ended January 11, 2006.
</item 2>

<item 3>
Sheila Carter is a continuing character on *The Young and the Restless*, first appearing in May 1990. Kimberlin Brown has played her since 1990, with Michelle
Stafford portraying her in 2006–07. Sheila has appeared in several stints and crossed over to *The Bold and the Beautiful*. Known by numerous aliases, she has
worked as a waitress, executive assistant, and nurse. Her family includes mother Molly, sister Sarah Smythe, and husband Scott Grainger Sr
</item 3>

<item 4>
Sheila Carter
She has also had conflicts with Stephanie Forrester , Maggie Forrester , Amber Moore , Taylor Hayes , Brooke Logan , Phyllis Summers and Quinn Fuller . Contents
( hide ) 1 Casting and creation 2 Development 3 Storylines 3.1 1990 -- 92 3.2 1992 -- 98 3.3 2002 -- 03 3.4 2005 -- 07 3.5 2010 , 2012 3.6 2017 -- 4 Critical
reception 5 References 6 External links Casting and creation ( edit ) The role was portrayed by Kimberlin Brown for a total of 16 years . Her first run was on
The Young and the Restless from May 1990 to May 20 , 1992 , when the character crossed over to The Bold and the Beautiful from May 21 , 1992 , to October 2 ,
1998 , again from May 24 to November 5 , 2002 , and again finally from September 26 to October 10 , 2003 .
</item 4>

<item 5>
Sheila Carter
( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter
Granddaughters Lucy Romalotti Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on
the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed
the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like
Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 . Sheila is known as a villain . A significant
portion of Sheila 's history on both soaps revolves around her long - running conflict with Lauren Fenmore , whom she has attempted to kill on multiple
occasions .
</item 5>

<item 6>
Sheila Carter is a villain in CBS’s *The Young and the Restless* and *The Bold and the Beautiful*, created by William J. Bell and introduced in 1990. Kimberlin
Brown played her in multiple runs through 2006; Michelle Stafford took over after Sheila’s plastic surgery made her resemble Phyllis Summers. Brown returned in
2017. Sheila’s long-running feud with Lauren Fenmore includes repeated attempts to kill her; she has also clashed with several Forrester
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
Sheila continues to insinuate herself into Eric 's life , faking a concussion after a catfight with Quinn to manipulate Eric into letting her stay in the
Forrester mansion . Sheila blackmails her ex-husband , James Warwick , into lying about the severity of her condition . Critical reception ( edit ) Sheila is
often considered to be one of the most popular villains in daytime . Kimberlin Brown was nominated for a Daytime Emmy in the role as Outstanding Supporting
Actress in 1993 . With the global success of The Bold and the Beautiful , Sheila is arguably the most recognisable soap opera villain in the world . References
( edit ) ^ Jump up to : `` About Y&R : Who 's Who in Los Angeles - Sheila Carter '' . SoapCental . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. (
9 July 2005 ) . `` Y&R confirms the return of Kimberlin Brown '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday
, 5th August 2005 '' . CBS.com .
</item 8>

<item 9>
Sheila Carter
. Daytime Confidential . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. ( 30 November 2009 ) . `` Is Kimberlin Brown returning to The Young and the
Restless ? '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ Logan , Michael ( June 9 , 2017 ) . `` Kimberlin Brown Returns to The Bold and the
Beautiful as Supervillain Sheila Carter '' . TV Insider . United States : CBS Interactive . Retrieved June 10 , 2017 . Jump up ^ Logan , Michael ( June 19 ,
2017 ) . `` The Bold and the Beautiful : EP Bradley Bell Answers Our Burning Sheila Carter Questions '' . TV Insider . United States : TVGM Holdings LLC .
Retrieved June 19 , 2017 . Jump up ^ Kimberlin Brown ( 1992 ) . The Bold and the Beautiful May 1992 Promo Sheila . CBS Daytime . Retrieved 7 November 2012 .
Jump up ^ Kimberlin Brown ( 2008 ) .
</item 9>

<item 10>
Sheila Carter
When asked where her alter - ego may be `` hiding out '' , Brown said : `` I am still getting healthy from my car accident . Right now , Sheila is hiding out in
North Carolina -- playing golf . But do n't worry : Sheila always comes back ! '' In June 2017 , Kimberlin Brown reprised her role as Sheila on The Bold and the
Beautiful . Executive producer Brad Bell announced Brown had been signed to a one - year deal with the serial . Development ( edit ) `` She seduced another
woman 's husband and stole their child ... but if you think that 's bold , hold on tight ... because the hand that rocked The Young and the Restless is now
rocking The Bold and the Beautiful . Sheila lives ... -- CBS promoting Kimberlin Brown 's crossover as Sheila On the topic of Sheila and Lauren 's `` cat fights
'' , Brown said : `` People like cat fights because we can get away with what you ca n't get away with in everyday life . You 're living vicariously through me
.
</item 10>
=== END ITEM id=c1f8f631c96b6d4d ===

=== ITEM id=744552870f36d1f9 ===
Gold sentences and facts:
Sentence 0: The culture of Australia is a Western culture , derived primarily from Britain but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people . The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government , Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement .
  Fact 0.0: The culture of Australia is a Western culture.
  Fact 0.1: The culture of Australia is derived primarily from Britain.
  Fact 0.2: The culture of Australia is also influenced by the unique geography of Australia.
  Fact 0.3: The culture of Australia is influenced by the diverse input of Aboriginal, Torres Strait Islander and other Australian people.
  Fact 0.4: British colonization of Australia began in 1788.
  Fact 0.5: Waves of multi-ethnic migration followed the British colonization of Australia.
  Fact 0.6: Evidence of a significant Anglo-Celtic heritage in Australia includes the predominance of the English language.
  Fact 0.7: Evidence of Anglo-Celtic heritage in Australia includes a democratic system of government drawing upon the British traditions of Westminster Government, Parliamentarianism and constitutional monarchy.
  Fact 0.8: Evidence of Anglo-Celtic heritage in Australia includes American constitutionalist and federalist traditions.
  Fact 0.9: Evidence of Anglo-Celtic heritage in Australia includes Christianity as the dominant religion.
  Fact 0.10: Evidence of Anglo-Celtic heritage in Australia includes the popularity of sports originating in (or influenced by) the British Isles.
  Fact 0.11: Australian culture has diverged significantly since British settlement.

Retrieved items:
<item 1>
Culture of Australia
Most Aboriginal Australians have a belief system based on the Dreaming , or Dreamtime , which refers both to a time when ancestral spirits created land and
culture , and to the knowledge and practices that define individual and community responsibilities and identity . Conflict and reconciliation between indigenous
and non-indigenous Australians has been a source of much art and literature in Australia , and ancient Aboriginal artistic styles and iconic inventions such as
the boomerang , the didgeridoo and Indigenous Australian music have become symbols of modern Australia . Governor Arthur Phillip hoists the British flag over
the new colony at Sydney in 1788 South Australian suffragette Catherine Helen Spence ( 1825 -- 1910 ) . The Australian colonies established democratic
parliaments from the 1850s and began to grant women the vote in the 1890s . The arrival of the first British settlers at what is now Sydney in 1788 introduced
Western civilisation to the Australian continent .
</item 1>

<item 2>
Culture of Australia
These wars profoundly affected Australia 's sense of nationhood and a proud military legend developed around the spirit of Australia 's ANZAC troops , who came
to symbolise the virtues of mateship , courage and endurance for the nation . The Australian colonies had a period of extensive multi-ethnic immigration during
the Australian gold rushes of the latter half of the 19th century , but following Federation in 1901 , the Parliament instigated the White Australia Policy that
gave preference to British migrants and ensured that Australia remained a predominantly Anglo - Celtic society until well into the 20th Century . The post-World
War II immigration program saw the policy dismantled by successive governments , permitting large numbers of Southern European , and later Asian and Middle
Eastern migrants to arrive . The Menzies Government ( 1949 - 1966 ) and Holt Government dismantled the legal barriers to multi-ethnic immigration and by the
1970s , the Whitlam and Fraser Governments were promoting multiculturalism . Countries of birth of Australian estimated resident population , 2006 Some States
and Territories of Australia retained discriminatory laws relating to voting rights for Aborigines into the 1960s , at which point full legal equality was
established .
</item 2>

<item 3>
Culture of Australia
The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes
the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government ,
Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the
popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement . Aboriginal
people are believed to have arrived as early as 60,000 years ago , and evidence of Aboriginal art in Australia dates back at least 30,000 years . Several states
and territories had their origins as penal colonies , with the first British convicts arriving at Sydney Cove in 1788 . Stories of outlaws like the bushranger
Ned Kelly have endured in Australian music , cinema and literature . The Australian gold rushes from the 1850s brought wealth as well as new social tensions to
Australia , including the miners ' Eureka Stockade rebellion . The colonies established elected parliaments and rights for workers and women before most other
Western nations .
</item 3>

<item 4>
Culture of Australia
A 1967 referendum to include all Aborigines in the national electoral roll census was overwhelmingly approved by voters . In 1984 , a group of Pintupi people
who were living a traditional hunter - gatherer desert - dwelling life were tracked down in the Gibson Desert and brought into a settlement . They are believed
to have been the last uncontacted tribe . While the British cultural influence remained strong into the 21st century , other influences became increasingly
important . The Hawaiian sport of surfing was adopted in Australia where a beach culture and the locally developed surf lifesaving movement was already
burgeoning in the early 20th century . American pop culture and cinema were embraced in the 20th century , with country music and later rock and roll sweeping
Australia , aided by the new technology of television and a host of American content . The 1956 Melbourne Olympics announced a confident , prosperous post-war
nation , and new cultural icons like Australian country music star Slim Dusty and dadaist Barry Humphries expressed a uniquely Australian identity . Australia
's contemporary immigration program has two components : a program for skilled and family migrants and a humanitarian program for refugees and asylum seekers .
</item 4>

<item 5>
Culture of Australia
By 2010 , the post-war immigration program had received more than 6.5 million migrants from every continent . The population tripled in the six decades to
around 21 million in 2010 , including people originating from 200 countries . More than 43 per cent of Australians were either born overseas or have one parent
who was born overseas . The population is highly urbanised , with more than 75 % of Australians living in urban centres , largely along the coast . Contemporary
Australia is a pluralistic society , rooted in liberal democratic traditions and espousing informality and egalitarianism as key societal values . While
strongly influenced by Anglo - Celtic origins , the culture of Australia has also been shaped by multi-ethnic migration which has influenced all aspects of
Australian life , including business , the arts , cuisine , sense of humour and sporting tastes . Contemporary Australia is also a culture that is profoundly
influenced by global movements of meaning and communication , including advertising culture . In turn , globalizing corporations from Holden to Exxon have
attempted to associate their brand with Australian cultural identity . This process intensified from the 1970s onwards .
</item 5>

<item 6>
Culture of Australia
Multicultural immigration has increased Australia 's religious diversity . The proportion of the total population who are Christian fell from 71 % in 1996 to
around 61.1 % in 2011 , while people affiliated with non-Christian religions increased from around 3.5 % to 7.2 % over the same period . Buddhism increased most
rapidly from 1.1 % to 2.5 % . Increased immigration from South - East Asia has been a major factor in this growth , but Australians of Anglo - Celtic origin
have also shown increasing interest in Buddhism . Islam increased during the period from 1.1 % to 2.2 % with diverse communities concentrated mainly in Sydney
and Melbourne . The history of the Jews in Australia dates back to the First Fleet , which brought Jewish convicts to Sydney in 1788 . Today , many Jews in
Australia originated as refugees and Holocaust survivors who arrived during and after World War II . Hindus came to Australia as labourers and merchants during
the 19th century and numbers increased dramatically from the 1960s , more than doubling between 1996 and 2006 .
</item 6>

<item 7>
Culture of Australia
Although Sydney was initially used by the British as a place of banishment for prisoners , the arrival of the British laid the foundations for Australia 's
democratic institutions and rule of law , and introduced the long traditions of English literature , Western art and music , and Judeo - Christian ethics and
religious outlook to a new continent . The British Empire expanded across the whole continent and established six colonies . The colonies were originally penal
colonies , with the exception of South Australia , which was established as a `` free colony '' with no convicts and a vision for a territory with political and
religious freedoms , together with opportunities for wealth through business and pastoral investments . Contact between the indigenous Australians and the new
settlers ranged from cordiality to violent conflict , but the diseases brought by Europeans were devastating to Aboriginal populations and culture . According
to the historian Geoffrey Blainey , during the colonial period : `` Smallpox , measles , influenza and other new diseases swept from one Aboriginal camp to
another ... The main conqueror of Aborigines was to be disease and its ally , demoralisation . '' William Wentworth established Australia 's first political
party in 1835 to demand democratic government for New South Wales .
</item 7>

<item 8>
Culture of Australia
Culture of Australia - Wikipedia Culture of Australia Jump to : navigation , search Part of a series on the Culture of Australia History People ( show )
Australian people African Australian Albanian Australian American Australian Anglo - Celtic Australian Asian Australian Assyrian Australian Bosnian Australian
Chinese Australian Cornish Australian English Australian European Australian Filipino Australian Greek Australian Indian Australian Indigenous Australian Irish
Australian Italian Australian Japanese Australian Jewish Australian Korean Australian Latin American Australian Lebanese Australian Macedonian Australian
Pakistani Australian Scottish Australian Serbian Australian South African Australian Sudanese Australian Taiwanese Australian Welsh Australian Languages
Traditions ( show ) Humour Mythology and folklore ( show ) Mythology folklore Cuisine Festivals Religion ( show ) Aboriginal Christianity ( Roman Catholic
Anglican Uniting Church Presbyterian Baptist Lutheran Coptic Orthodox more ) Buddhism Islam Sikhism Judaism Hinduism Bahá'í Faith Irreligion Art ( show )
Architecture Literature ( show ) Comic books Music and performing arts ( show ) Music Performing arts Media ( show ) Radio Television Cinema Sport Monuments (
show ) World Heritage Sites Symbols ( show ) Flag Coat of arms Australia portal The culture of Australia is a Western culture , derived primarily from Britain
but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people .
</item 8>

<item 9>
Culture of Australia
1 Indigenous music 5.6. 2 Folk music and national songs 5.6. 3 Classical music 5.6. 4 Popular music 5.7 Dance 5.8 Television 6 Religion 7 Public holidays 8
Cuisine 8.1 Beverages 9 Clothing and apparel 10 Sport 10.1 Cricket 10.2 Football codes 10.3 Water sports 10.4 Other sports 11 Folklore 12 Attitudes , beliefs
and stereotypes 13 References Development of Australian culture ( edit ) Main article : History of Australia A Luritja man demonstrating method of attack with
boomerang under cover of shield ( 1920 ) The oldest surviving cultural traditions in Australia -- and some of the oldest surviving cultural traditions on earth
-- are those of Australia 's Aboriginal and Torres Strait Islander peoples . Their ancestors have inhabited Australia for between 40,000 and 60,000 years ,
living a hunter - gatherer lifestyle . In 2006 , the Indigenous population was estimated at 517,000 people , or 2.5 per cent of the total population .
</item 9>

<item 10>
Culture of Australia
The colonies established elected parliaments and rights for workers and women before most other Western nations . Federation in 1901 evidenced a growing sense
of national identity that had developed over the latter half of the 19th century , as seen in the works of the Heidelberg School painters and writers like Banjo
Paterson , Henry Lawson and Dorothea Mackellar . The World Wars profoundly altered Australia 's sense of identity , with World War I introducing the ANZAC
legend , and World War II seeing a reorientation from Britain to the United States as the nation 's foremost major ally . After the second war , 6.5 million
migrants from 200 nations brought immense new diversity , and Australians grew increasingly aware of their proximity to Asia . Over time , the diverse food ,
lifestyle and cultural practices of immigrants have been absorbed into mainstream Australian culture . Contents ( hide ) 1 Development of Australian culture 2
Symbols 3 Language 4 Humour 5 Arts 5.1 Literature 5.2 Theatre 5.3 Architecture 5.4 Visual arts 5.5 Cinema 5.6 Music 5.6. 1 Indigenous music 5.6.
</item 10>
=== END ITEM id=744552870f36d1f9 ===

=== ITEM id=2f4a5a79395b28c3 ===
Gold sentences and facts:
Sentence 0: In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript 3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder ( built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of components that included charting , advanced UI , and data services ( Flex Data Services ) .
  Fact 0.0: In 2007, Adobe's first version release was Adobe Flash CS3 Professional.
  Fact 0.1: Adobe Flash CS3 Professional was the ninth major version of Flash.
  Fact 0.2: Adobe Flash CS3 Professional introduced the ActionScript 3.0 programming language.
  Fact 0.3: ActionScript 3.0 supported modern programming practices.
  Fact 0.4: ActionScript 3.0 enabled business applications to be developed with Flash.
  Fact 0.5: Adobe Flex Builder was built on Eclipse.
  Fact 0.6: Adobe Flex Builder targeted the enterprise application development market.
  Fact 0.7: Adobe Flex Builder was released in 2007.
  Fact 0.8: Flex Builder included the Flex SDK.
  Fact 0.9: The Flex SDK is a set of components that included charting, advanced UI, and data services.
  Fact 0.10: The data services in the Flex SDK are called Flex Data Services.

Retrieved items:
<item 1>
Adobe Flash
FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker Photoshop Photoshop Lightroom Premiere Pro Reader
Shockwave Player Server LiveCycle Flash Media Server BlazeDS ColdFusion JRun PhoneGap Build Technologies ActionScript Adobe AIR Adobe Flash Adobe Flex Adobe
Shockwave Adobe Font Folio Digital Negative ( DNG ) Authorware FlashPaper Portable Document Format ( PDF ) PostScript Web services Acrobat.com Adobe Solutions
Network Adobe Photoshop Express Adobe Premiere Express Adobe Spark Adobe Stock Acquisitions Aldus Macromedia Scene7 Omniture Behance Fotolia People Bruce Chizen
Charles Geschke Shantanu Narayen John Warnock Del Yocam Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Adobe_Flash&oldid=821675124 ''
Categories : Adobe Flash American inventions Cross-platform software Computing platforms C++ software Mac OS multimedia software MacOS multimedia software
Windows multimedia software Macromedia software Adobe software 1993 software Video game development software Hidden categories : Webarchive template wayback
links Pages using web citations with no URL Pages using citations with accessdate and no URL Use mdy dates from April 2012 Articles containing potentially dated
statements from February 2009 All articles containing potentially dated statements Articles containing potentially
</item 1>

<item 2>
Adobe Flash
Adobe Flash - Wikipedia Adobe Flash Jump to : navigation , search For the application / animation - maker , see Adobe Animate . For other uses , see Adobe Flash
( disambiguation ) . Adobe Flash Developer ( s ) Adobe Systems Macromedia FutureWave Target platform ( s ) Web browsers , iOS , Android , Windows , macOS Editor
software Adobe Animate , Flash Builder , FlashDevelop , Powerflasher FDT , Flash Catalyst Scaleform Player software Flash Player Adobe AIR OpenFL Scaleform
Gnash Lightspark Format ( s ) SWF FLV FLA Programming language ( s ) ActionScript Application ( s ) Animation Browser games Rich Internet applications Desktop
applications Mobile apps Mobile games console games computer games Status Active ( Until 2020 ) License Proprietary Adobe Flash is a multimedia software
platform used for production of animations , rich Internet applications , desktop applications , mobile applications , mobile games and embedded web browser
video players . Flash displays text , vector graphics and raster graphics to provide animations , video games and applications . It allows streaming of audio
and video , and can capture mouse , keyboard , microphone and camera input . Artists may produce Flash graphics and animations using Adobe Animate .
</item 2>

<item 3>
Adobe Flash
, blend modes ( similar to Adobe Photoshop ) , and advanced features for FLV video . Adobe ( edit ) Macromedia was acquired by Adobe Systems in 2005 , and the
entire Macromedia product line including Flash , Dreamweaver , Director / Shockwave , Fireworks ( which has since been discontinued ) and Authorware is now
handled by Adobe . In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript
3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder (
built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of
components that included charting , advanced UI , and data services ( Flex Data Services ) . In 2008 , Adobe released the tenth version of Flash , Adobe Flash
CS4 . Flash 10 improved animation capabilities within the Flash editor , adding a motion editor panel ( similar to Adobe After Effects ) , inverse kinematics (
bones ) , basic 3D object animation , object - based animation , and other text and graphics features .
</item 3>

<item 4>
Adobe Flash
Artists may produce Flash graphics and animations using Adobe Animate . Software developers may produce applications and video games using Adobe Flash Builder ,
FlashDevelop , Flash Catalyst , or any text editor when used with the Apache Flex SDK . End - users can view Flash content via Flash Player ( for web browsers )
, AIR ( for desktop or mobile apps ) or third - party players such as Scaleform ( for video games ) . Adobe Flash Player ( supported on Microsoft Windows ,
macOS and Linux ) enables end - users to view Flash content using web browsers . Adobe Flash Lite enabled viewing Flash content on older smartphones , but has
been discontinued and superseded by Adobe AIR . The ActionScript programming language allows the development of interactive animations , video games , web
applications , desktop applications and mobile applications . Programmers can implement Flash software using an IDE such as Adobe Animate , Adobe Flash Builder
, Adobe Director , FlashDevelop and Powerflasher FDT . Adobe AIR enables full - featured desktop and mobile applications to be developed with Flash and
published for Windows , macOS , Android , iOS , Xbox One , PlayStation 4 , and Nintendo Wii U and Switch .
</item 4>

<item 5>
Adobe Flash was a proprietary multimedia platform for browser-based animation, video, games, and interactive or desktop/mobile applications, using formats such
as SWF, FLV, and FLA and the ActionScript language. Macromedia developed Flash before Adobe acquired it in 2005; Adobe’s releases added ActionScript 3 and
improved animation tools, including 3D and inverse-kinematics features. Adobe Animate became the main authoring tool, while numerous third-party programs could
export SWF.
</item 5>

<item 6>
Adobe Flash
The Apache Flex ActionScript 3.0 compiler generates SWF files from ActionScript 3 files . Flex was the primary ActionScript 3 compiler and was actively
developed by Adobe before it was donated to Apache Software Foundation in 2011 . Adobe Animate -- primarily used to design graphics and animation , but supports
ActionScript scripting and debugging . Adobe Flash Builder -- enterprise application development & debugging , contains the Flex SDK with UI and charting
components . Adobe Scout -- a visual profiler to optimize performance of Flash content . CrossBridge -- a free SDK to cross-compile C++ code to run in Flash
Player . Third - party tools ( edit ) Third - party development tools have been created to assist developers in creating software applications and video games
with Flash . FlashDevelop is a free and open - source Flash ActionScript IDE , which includes a project manager and debugger for building applications on Flash
Player and Adobe AIR . Powerflasher FDT is a commercial ActionScript IDE similar to FlashDevelop . Haxe is an open source , high - level object - oriented
programming language geared towards web - content creation that can compile SWF files from Haxe programs .
</item 6>

<item 7>
Adobe Flash
CreateJS is library that while available separately was also adopted by Adobe as a replacement for Wallaby in CS6 . Unlike Wallaby , which was a standalone
program , the `` Toolkit for CreateJS '' only works as a plug - in inside Flash Professional ; it generates output for the HTML5 canvas , animated with
JavaScript . Around December 2013 , the toolkit was integrated directly into Flash Professional CC . Criticisms ( edit ) Mobile support ( edit ) Websites built
with Adobe Flash will not function on most modern mobile devices running Google Android or iOS ( iPhone , iPad ) . The only alternative is using HTML5 and
responsive web design to build websites that support both desktop and mobile devices . However , Flash is still used to build mobile games using Adobe AIR .
Such games will not work in mobile web browsers , but must be installed via the appropriate app store . Vendor dependence ( edit ) See also : Vendor lock - in
The reliance on Adobe for decoding Flash makes its use on the World Wide Web a concern -- the completeness of its public specifications are debated , and no
complete implementation of Flash is publicly available in source code form with a license that permits reuse .
</item 7>

<item 8>
Adobe Flash
Adobe Wallaby Server - side Adobe Flash Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC
Scaleform GFx SWiSH Max Swift 3D Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash
controversy Flash animation films television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software
Macromedia Adobe Systems Adobe Creative Suite and Creative Cloud Current products Acrobat After Effects Animate Audition Bridge Dreamweaver Flash Builder
Illustrator InCopy InDesign Lightroom Media Encoder Muse Photoshop Prelude Premiere Pro Spark XD Creative Cloud services Behance Business Catalyst Story Typekit
Former products Contribute Device Central Edge Edge Animate Encore Flash Catalyst Fireworks GoLive ImageReady OnLocation Soundbooth Stock Photos Ultra Adobe
eLearning Suite Current products Captivate Flash Dreamweaver Photoshop Acrobat Presenter Soundbooth Shared applications Bridge Device Central Adobe Systems
Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code Edge
Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker
Photoshop
</item 8>

<item 9>
Adobe Flash
Previously , developers could not use the specification for making SWF - compatible players , but only for making SWF - exporting authoring software . The
specification still omits information on codecs such as Sorenson Spark , however . Animation tools ( edit ) Official tools ( edit ) Main article : Adobe Animate
The Adobe Animate authoring program is primarily used to design graphics and animation and publish the same for websites , web applications , and video games .
The program also offers limited support for audio and video embedding , and ActionScript scripting . Adobe released Adobe LiveMotion , designed to create
interactive animation content and export it to a variety of formats , including SWF . LiveMotion failed to gain any notable user base . In February 2003 ,
Macromedia purchased Presedia , which had developed a Flash authoring tool that automatically converted PowerPoint files into Flash . Macromedia subsequently
released the new product as Breeze , which included many new enhancements .
</item 9>

<item 10>
Adobe Flash Player
( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark Gameswf Tamarin Animation tools Adobe Animate Toon Boom Moho GoAnimate
Stencyl Programming tools FlashDevelop Powerflasher FDT Apache Flex Haxe CrossBridge Adobe Scout FlashFirebug Libraries Starling Framework Away3D Flare3D
Stage3D PureMVC Ming List of Flex frameworks Converters Adobe Flash Media Live Encoder SWFTools swfmill Google Swiffy Adobe Wallaby Server - side Adobe Flash
Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC Scaleform GFx SWiSH Max Swift 3D
Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash controversy Flash animation films
television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software Macromedia Adobe Systems Adobe
Systems Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code
Edge Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse
</item 10>
=== END ITEM id=2f4a5a79395b28c3 ===

=== ITEM id=57128ac515596b45 ===
Gold sentences and facts:
Sentence 0: The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of the Mahanadi Delta .
  Fact 0.0: The Eastern Coastal Plains refers to a wide stretch of landmass of India.
  Fact 0.1: The Eastern Coastal Plains lie between the Eastern Ghats and the Bay of Bengal.
  Fact 0.2: The Eastern Coastal Plains are wider and more leveled than the western coastal plains.
  Fact 0.3: The Eastern Coastal Plains stretch from Tamil Nadu in the south to West Bengal in the north.
  Fact 0.4: The Eastern Coastal Plains stretch through Andhra Pradesh and Odisha.
  Fact 0.5: Chilka Lake is a brackish water lake along the eastern coastal plain.
  Fact 0.6: Chilka Lake lies in the state of Odisha.
  Fact 0.7: Chilka Lake stretches to the south of the Mahanadi Delta.
Sentence 1: It is locally known as Utkal Plains in the Northern part between Cossye and Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains .
  Fact 1.0: The Eastern Coastal Plains are locally known as Utkal Plains in the northern part between the Cossye and Rushikulya Rivers.
  Fact 1.1: The Eastern Coastal Plains are known as Northern Circars in the central part between the Rushikulya and Krishna Rivers.
  Fact 1.2: The Eastern Coastal Plains are known as Coromandel Coast in the southern part from the south of the river Krishna to the southern tip of mainland India at Cape Comorin.
  Fact 1.3: The Coromandel Coast merges with the Western Coastal Plains at Cape Comorin.

Retrieved items:
<item 1>
Ring of Fire
Plateau Coromandel Coast Konkan False Divi Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen
Glacier Bay of Bengal Gulf of Khambhat Gulf of Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands
Andaman Islands Nicobar Islands Maldive Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater
Sunda Islands Lesser Sunda Islands Indonesian Archipelago Timor New Guinea Bonis Peninsula Papuan Peninsula Huon Peninsula Huon Gulf Bird 's Head Peninsula
Gazelle Peninsula Philippine Archipelago Luzon Visayas Mindanao Leyte Gulf Gulf of Thailand East Indies Nanyang Alpide belt Asia - Pacific Tropical Asia Ring of
Fire Regions of Europe North Nordic Northwestern Scandinavia Scandinavian Peninsula Fennoscandia Baltoscandia Sápmi West Nordic Baltic Baltic Sea Gulf of
Bothnia Gulf of Finland Iceland Faroe Islands East Danubian countries Prussia Galicia Volhynia Donbass Sloboda Ukraine Sambia Peninsula Amber Coast Curonian
Spit Izyum
</item 1>

<item 2>
Geography of Nigeria
Plains rise to the north of the valleys . To the southwest of the Niger there is `` rugged '' highland , and to the southeast of the Benue hills and mountains
are found all the way to the border with Cameroon . Coastal plains are found in both the southwest and the southeast . Niger Delta ( edit ) This section needs
expansion . You can help by adding to it . ( January 2011 ) The Niger Delta is located in the southern part of Nigeria . It is one of the world 's largest
arcuate fan - shaped river deltas . The riverine area of the Niger Delta is a coastal belt of swamps bordering the Atlantic ocean . The swamps are vegetated
tidal flats formed by a reticulate pattern of interconnected meandering creeks and distributaries of the River Niger . Vegetation of the Niger Delta ( edit )
The vegetation of the Niger Delta consist mainly of forest swamps . The forest are of two types , nearest the sea is a belt of saline / brackish Mangrove swamp
separated from the sea by sand beach ridges within the mangrove swamp . Numerous sandy islands occur with fresh water vegetation .
</item 2>

<item 3>
Sub-Saharan Africa
Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous forests
Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False Divi
Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay of Bengal Gulf of Khambhat Gulf of
Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman Islands Nicobar Islands Maldive
Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater Sunda Islands Lesser Sunda Islands
Indonesian Archipelago Timor New Guinea Bonis Peninsula Papuan Peninsula Huon Peninsula Huon Gulf Bird 's Head Peninsula Gazelle Peninsula Philippine
Archipelago Luzon Visayas Mindanao Leyte Gulf Gulf of Thailand East Indies Nanyang Alpide belt Asia - Pacific Tropical Asia Ring of Fire Regions of Europe North
Nordic Northwestern Scandinavia Scandinavian Peninsula
</item 3>

<item 4>
Eastern coastal plains
pp. 20 -- 21 . ISBN 8171725163 . Geography of India Climate Climate Climatic regions Geology Fossil Parks Geology of India Indian Plate Stones Landforms Beaches
Canals Desert Extreme points Glaciers Islands Lakes Mountains Rivers Valleys Volcanoes Waterfalls Plains Indo - Gangetic Eastern Coastal Western Coastal Regions
North India Northeast India East India South India West India Central India Subdivisions Autonomous administrative divisions States and territories Districts
Cities Towns Municipalities Environment Biogeographic classification Ecoregions Fauna Flora Forests Issues Protected Areas Wildlife This Indian location article
is a stub . You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Eastern_coastal_plains&oldid=812520937 ''
Categories : Plains of India Coasts of India India geography stubs Hidden categories : Articles needing additional references from May 2015 All articles needing
additional references India articles missing geocoordinate data All articles needing coordinates All stub articles Talk About Wikipedia हिन्दी Português தமிழ்
Edit links This page was last edited on 28 November 2017 , at 08 : 32 .
</item 4>

<item 5>
Sub-Saharan Africa
Arabian Peninsula Najd Hejaz Tihamah Eastern Arabia South Arabia Hadhramaut Arabian Peninsula coastal fog desert Tigris -- Euphrates Mesopotamia Upper
Mesopotamia Lower Mesopotamia Sawad Nineveh plains Akkad ( region ) Babylonia Canaan Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant
Southern Levant Transjordan Jordan Rift Valley Israel Levantine Sea Golan Heights Hula Valley Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai
Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater
Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt
South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir
Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal
Plains Meghalaya subtropical forests MENASA Lower
</item 5>

<item 6>
Ring of Fire
Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant Southern Levant Transjordan Jordan Rift Valley Levantine Sea Golan Heights Hula Valley
Gaza Strip West Bank Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria
Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland
Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats
Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus
Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous
forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False
Divi Point Hindi Belt Ladakh Aksai Chin
</item 6>

<item 7>
Eastern coastal plains
Eastern coastal plains - Wikipedia Eastern coastal plains Jump to : navigation , search This article relies largely or entirely on a single source . Relevant
discussion may be found on the talk page . Please help improve this article by introducing citations to additional sources . ( May 2015 ) View of Fields at
Biccavolu , Eastern coastal plains , Andhra Pradesh The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats
and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through
Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of
the Mahanadi Delta . Deltas of many of India 's rivers form a major portion of these plains . The Mahanadi , Godavari , Kaveri and Krishna rivers drain these
plains . The region receives both the Northeast & Southwest monsoon rains with its annual rainfall averaging between 1,000 and 3,000 mm ( 39 and 118 in ) .
</item 7>

<item 8>
Dead Sea
Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu
Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley
Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic
plains moist deciduous forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau
Coromandel Coast Konkan False Divi Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay
of Bengal Gulf of Khambhat Gulf of Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman
Islands Nicobar Islands Maldive Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater
</item 8>

<item 9>
Economy of India
It is upgrading its infrastructure . As of May 2014 , India had completed over 22,600 kilometres ( 14,000 mi ) of 4 - or 6 - lane highways , connecting most of
its major manufacturing , commercial and cultural centres . India 's road infrastructure carries 60 % of freight and 87 % of passenger traffic . Indian Railways
is the fourth - largest rail network in the world , with a track length of 114,500 kilometres ( 71,100 mi ) and 7,172 stations . This government - owned - and -
operated railway network carried an average of 23 million passengers a day , and over a billion tonnes of freight in 2013 . India has a coastline of 7,500
kilometres ( 4,700 mi ) with 13 major ports and 60 operational non-major ports , which together handle 95 % of the country 's external trade by volume and 70 %
by value ( most of the remainder handled by air ) . Nhava Sheva , Mumbai is the largest public port , while Mundra is the largest private sea port .
</item 9>

<item 10>
Eastern coastal plains
The width of the plains varies between 100 and 130 km ( 62 to 80 miles ) . It is locally known as Utkal Plains in the Northern part between Cossye and
Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of
river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains . See also ( edit ) Western Coastal Plains
References ( edit ) Jump up ^ Sr. Bimcy ; Sr. Sisily ; Charlotte . Bibliographic information . Scholar Publishing House . pp. 20 -- 21 . ISBN 8171725163 .
</item 10>
=== END ITEM id=57128ac515596b45 ===

=== ITEM id=e2860eb969a49431 ===
Gold sentences and facts:
Sentence 0: In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or qualitative considerations , rather than easily measurable and quantifiable factors .
  Fact 0.0: In negotiation theory, the best alternative to a negotiated agreement (BATNA) is the most advantageous course of action a party can take if negotiations fail and no agreement can be reached.
  Fact 0.1: BATNA is the key focus of a successful negotiator.
  Fact 0.2: BATNA is the driving force behind a successful negotiator.
  Fact 0.3: A party should generally not accept a resolution worse than its BATNA.
  Fact 0.4: Care should be taken to ensure that deals are accurately valued.
  Fact 0.5: Deal valuation should take into account all considerations.
  Fact 0.6: Relationship value is a consideration in valuing a deal.
  Fact 0.7: Time value of money is a consideration in valuing a deal.
  Fact 0.8: The likelihood that the other party will live up to its side of the bargain is a consideration in valuing a deal.
  Fact 0.9: The other considerations are often difficult to value.
  Fact 0.10: The other considerations are frequently based on uncertain or qualitative considerations.
  Fact 0.11: The other considerations are qualitative rather than easily measurable and quantifiable factors.

Retrieved items:
<item 1>
Best alternative to a negotiated agreement
History ( edit ) BATNA was developed by negotiation researchers Roger Fisher and William Ury of the Harvard Program on Negotiation ( PON ) , in their series of
books on Principled negotiation that started with Getting to YES , unwittingly duplicating the game theory concept of a disagreement point from bargaining
problems pioneered by Nobel Laureate John Forbes Nash decades earlier . A Nash Equilibrium is reached among a group of players when no player can benefit from
changing strategies if every other player sticks to their current strategy . For example , Amy and Phil are in Nash Equilibrium if Amy is making the best
decision she can , taking into account Phil 's decision , and Phil is making the best decision he can , taking into account Amy 's decision . Likewise , a group
of players are in Nash Equilibrium if each one is making the best decision that he or she can , taking into account the decisions of the others .
</item 1>

<item 2>
Best alternative to a negotiated agreement
See also ( edit ) Conflict resolution research Opportunity cost References ( edit ) Jump up ^ Best Alternative to a Negotiated Agreement Negotiation Experts
Jump up ^ Best Alternative to a Negotiated Agreement YArooms Jump up ^ Negotiation , Readings , Exercises and Cases , Roy J. Lewicki ^ Jump up to : Nolan -
Haley , Jaqueline M ( 2001 ) . Alternative Dispute Resolution in a Nutshell . Thomson West . pp. 39 -- 50 . ISBN 978 - 0 - 314 - 18014 - 8 . ^ Jump up to :
Fisher and Ury , Roger and William ( 2011 ) . Getting to YES . Penguin Books . pp. 1 -- 170 . ISBN 978 - 0 - 14 - 311875 - 6 . Jump up ^ Gulliver , P.H ( 1979 )
. Disputes and Negotiation : A Cross Culture Perspective . Academic Press . p. 287 . Jump up ^ Morris , Michael W. ; Gelfand , Michele J. ( 2004 ) .
</item 2>

<item 3>
Best alternative to a negotiated agreement
, a non-profit organization . About Wikipedia
</item 3>

<item 4>
Best alternative to a negotiated agreement
Retrieved 1 October 2012 . External links ( edit ) BATNA Explained by David Venter Alternative definition of BATNA Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Best_alternative_to_a_negotiated_agreement&oldid=819966880 '' Categories : Negotiation Dispute resolution Business
terms Hidden categories : Articles needing more detailed references Articles needing additional references from July 2013 All articles needing additional
references All articles lacking reliable references Articles lacking reliable references from July 2013 All articles with unsourced statements Articles with
unsourced statements from October 2007 Articles lacking reliable references from February 2015 Wikipedia articles needing page number citations from July 2013
Talk Contents About Wikipedia Deutsch Español Հայերեն Polski Edit links This page was last edited on 12 January 2018 , at 08 : 19 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization .
</item 4>

<item 5>
Best alternative to a negotiated agreement
Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding . According to Fisher and Ury , when hard negotiators meet
soft negotiators , the hard negotiators usually win their position , but at the cost of potentially damaging the long term relationship between the parties .
Attractive alternatives are needed to develop a strong BATNA . In the best - selling book Getting to YES : Negotiating Agreement Without Giving In , the authors
give 3 suggestions of how to accomplish this : Inventing a list of actions one might take if no agreement is reached Converting some of the more promising ideas
and transforming them into tangible and partial alternatives Selecting the alternative that sounds best In negotiations involving different cultures , all
parties need to account for cultural cognitive behaviors and should not let judgments and biases affect the negotiation . The individual should be separate from
the objective . The purpose here , as Gulliver mentions , is for negotiation parties to be aware . Preparation at all levels , including prejudice - free
thoughts , emotion - free behavior , bias - free behavior are helpful according to Morris and Gelfand .
</item 5>

<item 6>
Best alternative to a negotiated agreement
Contents ( hide ) 1 Overview 2 History 3 See also 4 References 5 External links Overview ( edit ) The BATNA is often seen by negotiators not as a safety net ,
but rather as a point of leverage in negotiations . Although a negotiator 's alternative options should , in theory , be straightforward to evaluate , the
effort to understand which alternative represents a party 's BATNA is often not invested . Options need to be actual and actionable to be of value , however
without the investment of time , options will frequently be included that fail on one of these criteria . Most managers overestimate their BATNA whilst
simultaneously investing too little time into researching their real options . This can result in poor or faulty decision making and negotiating result .
Negotiators also need to be aware of the other negotiator 's BATNA and to identify how it compares to what they are offering . Some people may adopt aggressive
, coercive , threatening and / or deceptive techniques , this is known as a hard negotiation style ; a theoretical example of this is adversarial approach style
negotiation . Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding .
</item 6>

<item 7>
Bond (finance)
An alternative process for bond issuance , which is commonly used for smaller issues and avoids this cost , is the private placement bond . Bonds sold directly
to buyers may not be tradeable in the bond market . Historically an alternative practice of issuance was for the borrowing government authority to issue bonds
over a period of time , usually at a fixed price , with volumes sold on a particular day dependent on market conditions . This was called a tap issue or bond
tap . Features ( edit ) Principal ( edit ) 1978 $1,000 U.S. Treasury Bond Nominal , principal , par , or face amount is the amount on which the issuer pays
interest , and which , most commonly , has to be repaid at the end of the term . Some structured bonds can have a redemption amount which is different from the
face amount and can be linked to performance of particular assets . Maturity ( edit ) The issuer has to repay the nominal amount on the maturity date . As long
as all due payments have been made , the issuer has no further obligations to the bond holders after the maturity date .
</item 7>

<item 8>
Reinforcement
In 1970 , Baer and Wolf created a name for the use of natural reinforcers called `` behavior traps '' . A behavior trap requires only a simple response to enter
the trap , yet once entered , the trap can not be resisted in creating general behavior change . It is the use of a behavioral trap that increases a person 's
repertoire , by exposing them to the naturally occurring reinforcement of that behavior . Behavior traps have four characteristics : They are `` baited '' with
virtually irresistible reinforcers that `` lure '' the student to the trap Only a low - effort response already in the repertoire is necessary to enter the trap
Interrelated contingencies of reinforcement inside the trap motivate the person to acquire , extend , and maintain targeted academic / social skills They can
remain effective for long periods of time because the person shows few , if any , satiation effects As can be seen from the above , artificial reinforcement is
in fact created to build or develop skills , and to generalize , it is important that either a behavior trap is introduced to `` capture '' the skill and
utilize naturally occurring reinforcement to maintain or increase it .
</item 8>

<item 9>
Best alternative to a negotiated agreement
( 2004 ) . `` Cultural Differences and Cognitive Dynamics : Expanding the Cognitive Perspective on Negotiation '' . In Gelfand , Michele J. ; Brett , Jeanne M.
The Handbook of Negotiation and Culture . Stanford University Press . pp. 45 -- 70 . ISBN 978 - 0 - 8047 - 4586 - 4 . Jump up ^ Myerson , Roger B. `` Nash
Equilibrium and the History of Economic Theory '' ( PDF ) . Nash Equilibrium and the History of Economic Theory . Journal of Economic Literature . Retrieved 1
October 2012 . Jump up ^ Hawkins and Steiner , Jeff and Neil . `` The Nash Equilibrium Meets Batna '' ( PDF ) . Gamed Therory Varied Uses in ADR . Harvard
University Press . Archived from the original ( PDF ) on February 28 , 2013 . Retrieved 1 October 2012 .
</item 9>

<item 10>
Best alternative to a negotiated agreement
Best alternative to a negotiated agreement - wikipedia Best alternative to a negotiated agreement Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
July 2013 ) ( Learn how and when to remove this template message ) In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most
advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving
force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that
deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party
will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or
qualitative considerations , rather than easily measurable and quantifiable factors .
</item 10>
=== END ITEM id=e2860eb969a49431 ===
