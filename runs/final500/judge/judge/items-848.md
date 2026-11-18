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

This file has 7 items.

=== ITEM id=04ce32b12f810675 ===
Gold sentences and facts:
Sentence 0: State and Territory Capitals of Australia Jurisdiction Capital City Population State / Territory Population Date of statehood Capital since Image Australia and Australian Capital Territory Canberra 435,019 403,468 Must remain a territory as per s. 125 of the Australian Constitution 1913 View of Civic from Mount Ainslie , with Telstra Tower in the background New South Wales Sydney 5,029,768 7,759,274 1788 1788 Skyline of the Sydney central business district Victoria Melbourne 4,725,316 6,179,249 1851 1851 The Melbourne skyline on the Yarra River Queensland Brisbane 2,360,241 4,848,877 1859 1860 Brisbane 's CBD , with the Story Bridge in the foreground Western Australia Perth 2,022,044 2,558,951 1829 1829 View of Perth CBD skyline South Australia Adelaide 1,324,279 1,713,054 1836 1836 View of Adelaide Skyline . Tasmania Hobart 224,462 517,588 1825 1826 View of the Hobart downtown district and Mount Wellington from Constitution Dock Northern Territory Darwin 145,916 245,740 Has not attained Statehood 1911 View of Darwin CBD
  Fact 0.0: Canberra is the capital city of the Australian Capital Territory.
  Fact 0.1: Canberra has a population of 435,019.
  Fact 0.2: The Australian Capital Territory has a population of 403,468.
  Fact 0.3: The Australian Capital Territory must remain a territory as per s. 125 of the Australian Constitution.
  Fact 0.4: Canberra has been the capital since 1913.
  Fact 0.5: There is an image of a view of Civic from Mount Ainslie, with Telstra Tower in the background, for Canberra.
  Fact 0.6: Sydney is the capital city of New South Wales.
  Fact 0.7: Sydney has a population of 5,029,768.
  Fact 0.8: New South Wales has a population of 7,759,274.
  Fact 0.9: New South Wales attained statehood in 1788.
  Fact 0.10: Sydney has been the capital since 1788.
  Fact 0.11: There is an image of the Sydney central business district skyline for Sydney.
  Fact 0.12: Melbourne is the capital city of Victoria.
  Fact 0.13: Melbourne has a population of 4,725,316.
  Fact 0.14: Victoria has a population of 6,179,249.
  Fact 0.15: Victoria attained statehood in 1851.
  Fact 0.16: Melbourne has been the capital since 1851.
  Fact 0.17: There is an image of the Melbourne skyline on the Yarra River for Melbourne.
  Fact 0.18: Brisbane is the capital city of Queensland.
  Fact 0.19: Brisbane has a population of 2,360,241.
  Fact 0.20: Queensland has a population of 4,848,877.
  Fact 0.21: Queensland attained statehood in 1859.
  Fact 0.22: Brisbane has been the capital since 1860.
  Fact 0.23: There is an image of Brisbane's CBD, with the Story Bridge in the foreground, for Brisbane.
  Fact 0.24: Perth is the capital city of Western Australia.
  Fact 0.25: Perth has a population of 2,022,044.
  Fact 0.26: Western Australia has a population of 2,558,951.
  Fact 0.27: Western Australia attained statehood in 1829.
  Fact 0.28: Perth has been the capital since 1829.
  Fact 0.29: There is an image of the Perth CBD skyline for Perth.
  Fact 0.30: Adelaide is the capital city of South Australia.
  Fact 0.31: Adelaide has a population of 1,324,279.
  Fact 0.32: South Australia has a population of 1,713,054.
  Fact 0.33: South Australia attained statehood in 1836.
  Fact 0.34: Adelaide has been the capital since 1836.
  Fact 0.35: There is an image of the Adelaide skyline for Adelaide.
  Fact 0.36: Hobart is the capital city of Tasmania.
  Fact 0.37: Hobart has a population of 224,462.
  Fact 0.38: Tasmania has a population of 517,588.
  Fact 0.39: Tasmania attained statehood in 1825.
  Fact 0.40: Hobart has been the capital since 1826.
  Fact 0.41: There is an image of the Hobart downtown district and Mount Wellington from Constitution Dock for Hobart.
  Fact 0.42: Darwin is the capital city of the Northern Territory.
  Fact 0.43: Darwin has a population of 145,916.
  Fact 0.44: The Northern Territory has a population of 245,740.
  Fact 0.45: The Northern Territory has not attained statehood.
  Fact 0.46: Darwin has been the capital since 1911.
  Fact 0.47: There is an image of the Darwin CBD for Darwin.

Retrieved items:
<item 1>
List of Australian capital cities
Jump up ^ in 1911 , South Australia transferred responsibility for governing the Northern Territory to the Commonwealth Government of Australia . It was granted
self - governance in 1978 , however the Northern Territory is still considered a territory under Commonwealth rule , not a state . References ( edit ) Jump up ^
Norfolk Island - Kingston http://www.pitcairners.org/vt_kingston.html Norfolk Island 's Home on the Web Jump up ^
http://www.abs.gov.au/AUSSTATS/abs@.nsf/Lookup/3218.0Main+Features12016?OpenDocument Jump up ^ http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/3218.
</item 1>

<item 2>
Australia has eight state and territory capitals: Sydney (NSW), Melbourne (Victoria), Brisbane (Queensland), Perth (Western Australia), Adelaide (South
Australia), Hobart (Tasmania), Darwin (Northern Territory), and Canberra (ACT). Melbourne served as the national capital after Federation in 1901; the
government moved to Canberra in 1913. Each capital is its jurisdiction’s seat of government and, for states and internal territories, its most populous city.
The ACT and Northern Territory
</item 2>

<item 3>
Foreign relations of Australia
Public Service Women Government leaders Government ministers Economy Agriculture Dollar ( currency ) Energy Gross state product Home ownership Public housing
Household income Manufacturing Media Mining Poverty Reserve Bank Stock Exchange Superannuation Taxation Telecommunications Tourism Transportation Welfare system
Society Crime Demographics Education Ethnic groups Healthcare Homelessness Immigration Languages Religion Women Culture Architecture Art Arts Australian English
Cinema Cuisine Dance Literature Music Public holidays Radio Sport Symbols Television Theatre Outline Index Book Portal Politics of Australia Commonwealth
Monarch Governor - General Prime Minister Cabinet ( Shadow Cabinet ) Executive Council Ministry Foreign relations Parliament House of Representatives Senate
Opposition Leader High Court Lower courts Constitution Statute of Westminster Australia Act Federal elections pre-1969 1969 1972 1977 1980 1984 1993 1998 2007
2013 2016 next by - elections State / Territory governments Governors and Administrators Premiers and Chief Ministers Parliaments and Assemblies NSW Vic Qld WA
SA Tas ACT NT State / Territory elections Vic 2014 SA 2014 Tas 2014 NSW 2015 ACT 2016 NT 2016 WA 2017 Qld 2017 Vic 2018 SA 2018 Tas 2018 NSW 2019 ACT 2020 NT
2020 WA 2021
</item 3>

<item 4>
Australia’s 1912 Coat of Arms features the six state badges, a seven-pointed star representing the states and territories, and a kangaroo and emu. Green and
gold became national colours in 1984; the golden wattle became the floral emblem in 1988. Royal symbols remain prominent: the monarch appears on currency, about
12% of public land is Crown land, and many places and organisations bear royal names or insignia. Australia has no official language, but English is
</item 4>

<item 5>
List of Australian capital cities
WESTERN AUSTRALIA NORTHERN TERRITORY SOUTH AUSTRALIA QUEENSLAND NEW SOUTH WALES VICTORIA TASMANIA Perth Adelaide Melbourne Canberra ( ACT ) Sydney Brisbane
Hobart Darwin State and Territory Capitals of Australia Jurisdiction Capital City Population State / Territory Population Date of statehood Capital since Image
Australia and Australian Capital Territory Canberra 435,019 403,468 Must remain a territory as per s.
</item 5>

<item 6>
Culture of Australia
Subdivisions States and Territories Capitals Cities Politics Constitution Courts Elections Donations Electoral system Parties Federal budget Foreign relations
Government Governor - General Human rights Intersex LGBT Transgender Intelligence and security Law Constitution Judiciary Enforcement Military Monarchy National
debt Parliament Public Service Women Government leaders Government ministers Economy Agriculture Dollar ( currency ) Energy Gross state product Home ownership
Public housing Household income Manufacturing Media Mining Poverty Reserve Bank Stock Exchange Superannuation Taxation Telecommunications Tourism Transportation
Welfare system Society Crime Demographics Education Ethnic groups Healthcare Homelessness Immigration Languages Religion Women Culture Architecture Art Arts
Australian English Cinema Cuisine Dance Literature Music Public holidays Radio Sport Symbols Television Theatre Outline Index Book Portal Culture of Oceania
Sovereign states Australia Federated States of Micronesia Fiji Kiribati Marshall Islands Nauru New Zealand Palau Papua New Guinea Samoa Solomon Islands Tonga
Tuvalu Vanuatu Associated states of New Zealand Cook Islands Niue Dependencies and other territories American Samoa Christmas Island Cocos ( Keeling ) Islands
Easter Island French Polynesia Guam Hawaii New Caledonia Norfolk Island Northern Mariana Islands Pitcairn Islands Tokelau Wallis and Futuna Retrieved from ``
https://en.wikipedia.org/w/index.
</item 6>

<item 7>
List of Australian capital cities
wikipedia.org/w/index.php?title=List_of_Australian_capital_cities&oldid=813441772 '' Categories : Australian capital cities Demographics of Australia Lists of
capitals of country subdivisions Lists of cities in Australia Talk About Wikipedia Languages Bahasa Indonesia ქართული Македонски Bahasa Melayu Norsk Português
Русский தமிழ் اردو Edit links This page was last edited on 3 December 2017 , at 19 : 08 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
List of Australian capital cities
02016?OpenDocument External links ( edit ) Capital cities of Australia National and ACT Canberra NSW Sydney NT Darwin QLD Brisbane SA Adelaide TAS Hobart VIC
Melbourne WA Perth List of cities in Australia Australia articles History Timeline Bibliography Prehistory Archaeology European Exploration Territorial
evolution 1788 -- 1850 1851 -- 1900 1901 -- 1945 Since 1945 By topic Constitutional Diplomatic Economic Federation Immigration Indigenous Military Monarchy
Railway Geography Climate Continent Deserts Environment issues Earthquakes Flora Fauna Forests Geology Islands Mountains Regions Protected areas Rivers
Subdivisions States and Territories Capitals Cities Politics Constitution Courts Elections Donations Electoral system Parties Federal budget Foreign relations
Government Governor - General Human rights Intersex LGBT Transgender Intelligence and security Law Constitution Judiciary Enforcement Military Monarchy National
debt Parliament Public Service Women Government leaders Government ministers Economy Agriculture Dollar ( currency ) Energy Gross state product Home ownership
Public housing Household income Manufacturing Media Mining Poverty Reserve Bank Stock Exchange Superannuation Taxation Telecommunications Tourism Transportation
Welfare system Society Crime Demographics Education Ethnic groups Healthcare Homelessness Immigration Languages Religion Women Culture Architecture Art Arts
Australian English Cinema Cuisine Dance Literature Music Public holidays Radio Sport Symbols Television Theatre Outline Index Book Portal Retrieved from ``
https://en.wikipedia.
</item 8>

<item 9>
List of Australian capital cities
List of Australian capital cities - wikipedia List of Australian capital cities Jump to : navigation , search There are eight capital cities in Australia , each
of which function as the seat of government for the state or territory in which they are located . Melbourne was the initial capital following the 1901
Federation of Australia . In 1913 , the seat of national government was moved to the newly created city of Canberra , which continues to serve as the national
capital . Each capital city hosts the judicial , administrative and legislative functions for their jurisdiction . In the case of the states and internal
territories , their capital cities also happen to be the most populous city in their respective jurisdiction . The Australian external territory of Norfolk
Island has its official capital at Kingston , although this acts merely as the administrative centre of government ; its de facto capital is Burnt Pine .
</item 9>

<item 10>
List of Australian capital cities
125 of the Australian Constitution 1913 View of Civic from Mount Ainslie , with Telstra Tower in the background New South Wales Sydney 5,029,768 7,759,274 1788
1788 Skyline of the Sydney central business district Victoria Melbourne 4,725,316 6,179,249 1851 1851 The Melbourne skyline on the Yarra River Queensland
Brisbane 2,360,241 4,848,877 1859 1860 Brisbane 's CBD , with the Story Bridge in the foreground Western Australia Perth 2,022,044 2,558,951 1829 1829 View of
Perth CBD skyline South Australia Adelaide 1,324,279 1,713,054 1836 1836 View of Adelaide Skyline . Tasmania Hobart 224,462 517,588 1825 1826 View of the Hobart
downtown district and Mount Wellington from Constitution Dock Northern Territory Darwin 145,916 245,740 Has not attained Statehood 1911 View of Darwin CBD Notes
( edit ) Jump up ^ Although the ACT has its own government , it remains a territory ultimately under Commonwealth rule .
</item 10>
=== END ITEM id=04ce32b12f810675 ===

=== ITEM id=4c932f44905d6761 ===
Gold sentences and facts:
Sentence 0: Only Mexico , Italy , France , Germany ( West Germany until shortly after the 1990 World Cup ) and Brazil have hosted the event on two occasions . Mexico City 's Estadio Azteca and Rio de Janeiro 's Maracanã are the only venues ever to have hosted two FIFA World Cup finals . Only the 2002 FIFA World Cup had more than one host , being split between Japan and South Korea .
  Fact 0.0: Only Mexico has hosted the event on two occasions.
  Fact 0.1: Only Italy has hosted the event on two occasions.
  Fact 0.2: Only France has hosted the event on two occasions.
  Fact 0.3: Only Germany has hosted the event on two occasions.
  Fact 0.4: Only Brazil has hosted the event on two occasions.
  Fact 0.5: Germany's hosting covers West Germany until shortly after the 1990 World Cup.
  Fact 0.6: Mexico City's Estadio Azteca is one of the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.7: Rio de Janeiro's Maracanã is one of the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.8: Estadio Azteca and the Maracanã are the only venues ever to have hosted two FIFA World Cup finals.
  Fact 0.9: Only the 2002 FIFA World Cup had more than one host.
  Fact 0.10: The 2002 FIFA World Cup was split between Japan and South Korea.

Retrieved items:
<item 1>
FIFA World Cup hosts
33 1.3 12 Fourth place South Korea 2002 7 57 % 8 6 0.3 1.1 13 Fourth place Brazil 2014 7 57 % 11 14 − 3 − 0.4 1.57 14 Quarter - final Mexico 1986 5 0 80 % 6 0.8
15 Quarter - final Mexico 1970 63 % 6 0.5 1.5 16 Quarter - final Switzerland 1954 0 50 % 11 11 0 0 2.8 17 Quarter - final France 1938 0 50 % 0 0 18 Second round
Spain 1982 5 40 % 5 − 1 − 0.2 0.8 19 Round of 16 Japan 2002 63 % 5 0.5 1.25 20 Round of 16 United States 1994 38 % − 1 − 0.3 0.8 21 Group stage South Africa 50
% 5 − 2 − 0.
</item 1>

<item 2>
FIFA World Cup hosts
1962 , 1966 , 1974 , 1982 , 2006 Argentina 5 1938 , 1962 , 1970 , 1978 , 2030 Mexico 1970 , 1978 , 1986 , 2002 , 2026 Morocco 1994 , 1998 , 2006 , 2010 , 2026
Spain 1930 , 1966 , 1974 , 1982 , 2018 Brazil 1950 , 1994 , 2006 , 2014 England 1966 , 1990 , 2006 , 2018 United States 1986 , 1994 , 2022 , 2026 Italy 1930 ,
1934 , 1990 Japan 1970 , 2002 , 2022 Canada 1986 , 2026 Chile 1962 , 2014 Colombia 1986 , 2014 France 1938 , 1998 Netherlands 1930 , 2018 Russia 1990 , 2018
South Africa 2006 , 2010 South Korea 2002 , 2022 Switzerland 1954 ,
</item 2>

<item 3>
FIFA World Cup hosts
FIFA World Cup bids 1930 Hungary Italy Netherlands Spain Sweden Uruguay 1934 Italy Sweden 1938 Argentina France Germany 1942 Brazil Germany 1946 None 1950
Brazil 1954 Switzerland 1958 Sweden 1962 Argentina Chile West Germany 1966 England West Germany Spain 1970 Argentina Mexico Italy Netherlands Spain West Germany
1978 Argentina Mexico 1982 Spain West Germany 1986 Canada Colombia Mexico United States 1990 Austria England France Greece Italy Soviet Union West Germany
Yugoslavia 1994 Brazil Morocco United States 1998 France Morocco Switzerland 2002 Japan -- South Korea Mexico 2006 Brazil England Germany Morocco South Africa
Egypt Libya -- Tunisia Morocco Nigeria South Africa 2014 Brazil Colombia 2018 Belgium -- Netherlands England Portugal -- Spain Russia 2022 Australia Japan Qatar
South Korea United States 2026 Canada -- Mexico -- United States Morocco 2030 Argentina -- Paraguay -- Uruguay England Notes : Bold entries are those chosen as
hosts Event cancelled due to World War II Colombia withdrew after being award event , which was re-awarded to Mexico Retrieved from ``
https://en.wikipedia.org/w/index.
</item 3>

<item 4>
FIFA World Cup hosts
1 Accusations of bribery 21 2026 FIFA World Cup 22 2030 FIFA World Cup 23 2034 FIFA World Cup 24 Total bids by country 25 Host country performances 26
References 27 External links List of hosts ( edit ) Year Host Continent Winner 1930 Uruguay South America Uruguay 1934 Italy Europe Italy 1938 France Europe
Italy 1942 Cancelled due to World War II 1946 Cancelled due to World War II 1950 Brazil South America Uruguay 1954 Switzerland Europe West Germany 1958 Sweden
Europe Brazil 1962 Chile South America Brazil 1966 England Europe England 1970 Mexico North America Brazil West Germany Europe West Germany 1978 Argentina South
America Argentina 1982 Spain Europe Italy 1986 Mexico North America Argentina 1990 Italy Europe West Germany 1994 United States North America Brazil 1998 France
Europe France 2002 South Korea / Japan Asia Brazil 2006 Germany Europe Italy South Africa Africa Spain 2014 Brazil South America Germany 2018 Russia Europe 2022
Qatar Asia 2026 TBD 2030 TBD 1930 FIFA World Cup ( edit ) Main article : 1930 FIFA World Cup Bids : Hungary Italy
</item 4>

<item 5>
FIFA World Cup hosts
9 2.1 Champion England 1966 6 5 0 92 % 11 8 1.3 1.8 Champion Italy 1934 5 0 90 % 12 9 1.8 2.4 5 Champion West Germany 7 6 0 86 % 13 9 1.3 1.9 6 Champion
Argentina 1978 7 5 79 % 15 9 1.3 2.1 7 Runner - up Brazil 1950 6 75 % 22 6 16 2.7 3.7 8 Runner - up Sweden 1958 6 75 % 12 7 5 0.8 9 Third place Italy 1990 7 6 0
93 % 10 8 1.1 1.4 10 Third place Germany 2006 7 5 79 % 14 6 8 1.1 11 Third place Chile 1962 6 0 67 % 10 8 0.33 1.
</item 5>

<item 6>
FIFA World Cup hosts
Contents ( hide ) 1 List of hosts 2 1930 FIFA World Cup 3 1934 FIFA World Cup 4 1938 FIFA World Cup 5 Cancelled FIFA World Cups 1942 and 1946 6 1950 and 1954
FIFA World Cups 6.1 1949 bid 6.2 1951 bid 7 1958 FIFA World Cup 8 1962 FIFA World Cup 9 1966 FIFA World Cup 10 1970 FIFA World Cup 11 1974 , 1978 , 1982 FIFA
World Cups 11.1 1974 results 11.2 1978 results 11.3 1982 results 12 1986 FIFA World Cup 13 1990 FIFA World Cup 14 1994 FIFA World Cup 15 1998 FIFA World Cup 16
2002 FIFA World Cup 17 2006 FIFA World Cup 17.1 Controversy 18 2010 FIFA World Cup 18.1 Controversy 19 2014 FIFA World Cup 20 2018 and 2022 FIFA World Cups 20.1
Accusations of bribery 21
</item 6>

<item 7>
FIFA World Cup hosts
FIFA World Cup hosts - wikipedia FIFA World Cup hosts Jump to : navigation , search Map of FIFA World Cup final hosts , 1930 -- 2014 . Dark green : twice ;
middle green : once ; light green : planned until 2022 Sixteen countries have been FIFA World Cup hosts in the competition 's twenty tournaments since the
inaugural World Cup in 1930 . The organization at first awarded hosting to countries at meetings of FIFA 's congress . The choice of location was controversial
in the earliest tournaments , given the three - week boat journey between South America and Europe , the two centers of strength in football at the time . The
decision to hold the first cup in Uruguay , for example , led to only four European nations competing . The next two World Cups were both held in Europe . The
decision to hold the second of these , the 1938 FIFA World Cup , in France was controversial , as the South American countries had been led to understand that
the World Cup would rotate between the two continents . Both Argentina and Uruguay thus boycotted the tournament .
</item 7>

<item 8>
FIFA World Cup hosts
^ Jump up to : Joint bid by Libya and Tunisia . Host country performances ( edit ) Further information : National team appearances in the FIFA World Cup It is
widely considered that home advantage is common in the World Cup , with the host team usually performing above average . Of the eight countries that have won
the tournament , only Brazil and Spain were not champions at home , and both England and France had their only titles as hosts . Further , Sweden got to its
only final as hosts of the 1958 tournament , and only the Swedes and Brazilians finished as runner - up in home soil . The host country reached the semi-finals
thirteen times in twenty tournaments , and both Chile and South Korea had their only finishes on the top four at home . Only South Africa in 2010 managed to not
go past the group stage . # Performance Team Year Pld Win % GF GA GD GD / M GF / M Champion Uruguay 1930 0 0 100 % 15 12 3.8 Champion France 1998 7 6 0 93 % 15
13 1.9 2.
</item 8>

<item 9>
FIFA World Cup hosts
1938 Brazil 1950 Switzerland 1954 Sweden 1958 Chile 1962 England 1966 Mexico 1970 West Germany 1974 Argentina 1978 Spain 1982 Mexico 1986 Italy 1990 United
States 1994 France 1998 South Korea / Japan 2002 Germany 2006 South Africa 2010 Brazil 2014 Russia 2018 Qatar 2022 2026 2030 2034 Qualification 1930 1934 1938
1950 1954 1958 1962 1966 1970 1978 1982 1986 1990 1994 1998 2002 2006 2014 2018 Finals 1930 1934 1938 1950 1954 1958 1962 1966 1970 1978 1982 1986 1990 1994
1998 2002 2006 2014 2018 Squads 1930 1934 1938 1950 1954 1958 1962 1966 1970 1978 1982 1986
</item 9>

<item 10>
FIFA World Cup hosts
1962 1966 1970 1978 1982 1986 1990 1994 1998 2002 2006 2014 2018 Seedings 1998 2002 2006 2014 2018 Broadcasters 1998 2002 2006 2014 2018 Bids 2014 2018 and 2022
2026 2030 Records and statistics All - time table Goalscorers top goalscorers finals goalscorers hat - tricks own goals Penalty shoot - outs Player appearances
Red cards Referees Team appearances Teams with no appearances Miscellaneous Awards Balls Economics Final draw History Hosts Mascots Official films Official
songs Organisers Trophy Video games Notes : There was no qualification for the 1930 World Cup as places were given by invitation only . In 1950 , there was no
final ; the article is about the decisive match of the final group stage .
</item 10>
=== END ITEM id=4c932f44905d6761 ===

=== ITEM id=2f3e765cada53301 ===
Gold sentences and facts:
Sentence 0: Central London Property Trust Ltd v High Trees House Ltd ( 1947 ) KB 130 ( or the High Trees case ) is an English contract law decision in the High Court . It reaffirmed and extended the doctrine of promissory estoppel in contract law in England and Wales . However , the most significant part of the judgment is obiter dicta as it relates to hypothetical facts ; that is , the landlord did not seek repayment of the full wartime rent .
  Fact 0.0: Central London Property Trust Ltd v High Trees House Ltd is cited as (1947) KB 130.
  Fact 0.1: Central London Property Trust Ltd v High Trees House Ltd is also called the High Trees case.
  Fact 0.2: Central London Property Trust Ltd v High Trees House Ltd is an English contract law decision in the High Court.
  Fact 0.3: The decision reaffirmed the doctrine of promissory estoppel in contract law in England and Wales.
  Fact 0.4: The decision extended the doctrine of promissory estoppel in contract law in England and Wales.
  Fact 0.5: The most significant part of the judgment is obiter dicta.
  Fact 0.6: The obiter dicta relates to hypothetical facts.
  Fact 0.7: In the hypothetical facts, the landlord did not seek repayment of the full wartime rent.

Retrieved items:
<item 1>
Central London Property Trust Ltd v High Trees House Ltd
If I were to consider this matter without regard to recent developments in the law , there is no doubt that had the plaintiffs claimed it , they would have been
entitled to recover ground rent at the rate of £ 2,500 a year from the beginning of the term , since the lease under which it was payable was a lease under seal
which , according to the old common law , could not be varied by an agreement by parol ( whether in writing or not ) , but only by deed . Equity , however
stepped in , and said that if there has been a variation of a deed by a simple contract ( which in the case of a lease required to be in writing would have to
be evidenced by writing ) , the courts may give effect to it as is shown in Berry v. Berry ( 1929 ) 2 K.B. 316 . That equitable doctrine , however , could
hardly apply in the present case because the variation here might be said to have been made without consideration . With regard to estoppel , the representation
made in relation to reducing the rent was not a representation of an existing fact .
</item 1>

<item 2>
Central London Property Trust Ltd v High Trees House Ltd
for last two quarters of 1945 ) . Judgment ( edit ) Based on previous judgments as Hughes v Metropolitan Railway Co , Denning J held that the full rent was
payable from the time that the flats became fully occupied in mid-1945 . However , he continued in an obiter statement that if Central London had tried to claim
for the full rent from 1940 onwards , they would not have been able to . This was reasoned on the basis that if a party leads another party to believe that he
will not enforce his strict legal rights , then the Courts will prevent him from doing so at a later stage . Being obiter dicta and in a court of first instance
this was doubly not a binding precedent , yet it essentially created the doctrine of promissory estoppel .
</item 2>

<item 3>
Central London Property Trust Ltd v High Trees House Ltd
It was a representation , in effect , as to the future , namely , that payment of the rent would not be enforced at the full rate but only at the reduced rate .
Such a representation would not give rise to an estoppel , because , as was said in Jorden v. Money ( 1854 ) 5 H.L.C. 185 , a representation as to the future
must be embodied as a contract or be nothing . But what is the position in view of developments in the law in recent years ? The law has not been standing still
since Jorden v. Money ( 1854 ) 5 H.L.C. 185 . There has been a series of decisions over the last fifty years which , although they are said to be cases of
estoppel are not really such . They are cases in which a promise was made which was intended to create legal relations and which , to the knowledge of the
person making the promise , was going to be acted on by the person to whom it was made and which was in fact so acted on . In such cases the courts have said
that the promise must be honoured .
</item 3>

<item 4>
Central London Property Trust Ltd v High Trees House Ltd
I am satisfied that a promise such as that to which I have referred is binding and the only question remaining for my consideration is the scope of the promise
in the present case . I am satisfied on all the evidence that the promise here was that the ground rent should be reduced to £ 1,250 a year as a temporary
expedient while the block of flats was not fully , or substantially fully let , owing to the conditions prevailing . That means that the reduction in the rent
applied throughout the years down to the end of 1944 , but early in 1945 it is plain that the flats were fully let , and , indeed the rents received from them (
many of them not being affected by the Rent Restrictions Acts ) , were increased beyond the figure at which it was originally contemplated that they would be
let . At all events the rent from them must have been very considerable . I find that the conditions prevailing at the time when the reduction in rent was made
, had completely passed away by the early months of 1945 .
</item 4>

<item 5>
Central London Property Trust Ltd v High Trees House Ltd
I am satisfied that the promise was understood by all parties only to apply under the conditions prevailing at the time when it was made , namely , when the
flats were only partially let , and that it did not extend any further than that . When the flats became fully let , early in 1945 , the reduction ceased to
apply . In those circumstances , under the law as I hold it , it seems to me that rent is payable at the full rate for the quarters ending September 29 and
December 25 , 1945 . If the case had been one of estoppel , it might be said that in any event the estoppel would cease when the conditions to which the
representation applied came to an end , or it also might be said that it would only come to an end on notice . In either case it is only a way of ascertaining
what is the scope of the representation . I prefer to apply the principle that a promise intended to be binding , intended to be acted on and in fact acted on ,
is binding so far as its terms properly apply .
</item 5>

<item 6>
Central London Property Trust Ltd v High Trees House Ltd
In such cases the courts have said that the promise must be honoured . The cases to which I particularly desire to refer are : Fenner v. Blake ( 1900 ) 1 Q.B.
426 , In re Wickham ( 1917 ) 34 T.L.R. 158 , Re William Porter & Co. , Ld. ( 1937 ) 2 All E.R. 361 and Buttery v. Pickard ( 1946 ) W.N. 25 . As I have said they
are not cases of estoppel in the strict sense . They are really promises - promises intended to be binding , intended to be acted on , and in fact acted on .
Jorden v. Money ( 1854 ) 5 H.L.C. 185 can be distinguished , because there the promisor made it clear that she did not intend to be legally bound , whereas in
the cases to which I refer the proper inference was that the promisor did intend to be bound .
</item 6>

<item 7>
Central London Property Trust Ltd v High Trees House Ltd
Central London Property Trust Ltd v. High Trees House Ltd - wikipedia Central London Property Trust Ltd v. High Trees House Ltd Jump to : navigation , search
High Trees case Court King 's Bench Division Decided 18 July 1946 Citation ( s ) ( 1947 ) KB 130 ; ( 1956 ) 1 All ER 256 ; 62 TLR 557 ; ( 1947 ) LJR 77 ; 175 LT
333 Court membership Judge ( s ) sitting Denning J Keywords Promissory estoppel Central London Property Trust Ltd v High Trees House Ltd ( 1947 ) KB 130 ( or
the High Trees case ) is an English contract law decision in the High Court . It reaffirmed and extended the doctrine of promissory estoppel in contract law in
England and Wales . However , the most significant part of the judgment is obiter dicta as it relates to hypothetical facts ; that is , the landlord did not
seek repayment of the full wartime rent .
</item 7>

<item 8>
Central London Property Trust Ltd v High Trees House Ltd
Denning J held estoppel to be applicable if ... ... a promise was made which was intended to create legal relations and which , to the knowledge of the person
making the promise , was going to be acted on by the person to whom it was made and which was in fact so acted on . Contents ( hide ) 1 Facts 2 Judgment 3
Significance 4 See also 5 References Facts ( edit ) In 1937 , High Trees House Ltd leased a block of flats in Clapham , London , for a rate £ 2500 / year from
Central London Property Trust Ltd . Due to the conditions during the beginning of World War II occupancy rates were drastically lower than normal . In January
1940 , to ameliorate the situation , the parties made an agreement in writing to reduce rent by half . However , neither party stipulated the period for which
this reduced rental was to apply . Over the next five years , High Trees paid the reduced rate while the flats began to fill , and by 1945 , the flats were back
at full occupancy . Central London sued for payment of the full rental costs from June 1945 onwards ( i.e.
</item 8>

<item 9>
Central London Property Trust Ltd v High Trees House Ltd
Here it was binding as covering the period down to the early part of 1945 , and as from that time full rent is payable . I therefore give judgment for the
plaintiff company for the amount claimed . Significance ( edit ) Advances have been made in promissory estoppel since its inception in High Trees to create a
new inroad into the rule in Pinnel 's case that an agreement to accept part payment of a debt in full satisfaction of it is unenforceable for want of
consideration . Denning commented that such an agreement should now be enforceable under the doctrine of promissory estoppel , and indeed the plaintiff did not
seek the full debt on the basis of what was fair and , perhaps , thought was the law . However , the courts were at first reluctant to overrule or distinguish
cases like Pinnel 's case and Foakes v Beer having formed part of the common law for so long . Lady Justice Arden in Collier v P & MJ Wright ( Holdings ) Ltd (
2007 ) accepted in principle that High Trees could be used to extinguish a creditor 's right to full payment of a debt in such circumstances .
</item 9>

<item 10>
Central London Property Trust Ltd v High Trees House Ltd
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Central_London_Property_Trust_Ltd_v_High_Trees_House_Ltd&oldid=837737647 '' Categories : English
contract case law Lord Denning cases English enforceability case law English estoppel case law 1946 in British law High Court of Justice cases 1946 in case law
Hidden categories : Use British English from August 2015 Use dmy dates from February 2012 Talk Contents About Wikipedia Add links This page was last edited on
22 April 2018 , at 18 : 40 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site ,
you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 10>
=== END ITEM id=2f3e765cada53301 ===

=== ITEM id=b72e8a3ed44997b6 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 107 `` Watch Me '' Jann Turner Shonda Rhimes October 5 , 2017 ( 2017 - 10 - 05 ) 701 5.52 108 `` Pressing the Flesh '' Tony Goldwyn Matt Byrne October 12 , 2017 ( 2017 - 10 - 12 ) 702 5.00 109 `` Day 101 '' Scott Foley Zahir McGhee October 19 , 2017 ( 2017 - 10 - 19 ) 703 4.70 110 `` Lost Girls '' Nicole Rubio Ameni Rozsa & Austin Guzman October 26 , 2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 ) 705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 - 01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 , 2018 ( 2018 - 01 - 25 ) 708 TBD 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD 117 11 `` Army of One '' TBA TBA February 8 , 2018 ( 2018 - 02 - 08 ) 711 TBD 118 12 TBA TBA TBA March 1 , 2018 ( 2018 - 03 - 01 ) TBA TBD
  Fact 0.0: Episode 107 overall is titled 'Watch Me'.
  Fact 0.1: Episode 107 overall was directed by Jann Turner.
  Fact 0.2: Episode 107 overall was written by Shonda Rhimes.
  Fact 0.3: Episode 107 overall originally aired on October 5, 2017.
  Fact 0.4: Episode 107 overall has production code 701.
  Fact 0.5: Episode 107 overall had 5.52 million U.S. viewers.
  Fact 0.6: Episode 108 overall is titled 'Pressing the Flesh'.
  Fact 0.7: Episode 108 overall was directed by Tony Goldwyn.
  Fact 0.8: Episode 108 overall was written by Matt Byrne.
  Fact 0.9: Episode 108 overall originally aired on October 12, 2017.
  Fact 0.10: Episode 108 overall has production code 702.
  Fact 0.11: Episode 108 overall had 5.00 million U.S. viewers.
  Fact 0.12: Episode 109 overall is titled 'Day 101'.
  Fact 0.13: Episode 109 overall was directed by Scott Foley.
  Fact 0.14: Episode 109 overall was written by Zahir McGhee.
  Fact 0.15: Episode 109 overall originally aired on October 19, 2017.
  Fact 0.16: Episode 109 overall has production code 703.
  Fact 0.17: Episode 109 overall had 4.70 million U.S. viewers.
  Fact 0.18: Episode 110 overall is titled 'Lost Girls'.
  Fact 0.19: Episode 110 overall was directed by Nicole Rubio.
  Fact 0.20: Episode 110 overall was written by Ameni Rozsa and Austin Guzman.
  Fact 0.21: Episode 110 overall originally aired on October 26, 2017.
  Fact 0.22: Episode 110 overall has production code 704.
  Fact 0.23: Episode 110 overall had 4.88 million U.S. viewers.
  Fact 0.24: Episode 111 overall is episode 5 of its season and is titled 'Adventures in Babysitting'.
  Fact 0.25: Episode 111 overall was directed by Oliver Bokelberg.
  Fact 0.26: Episode 111 overall was written by Serveriano Canales and Tia Napolitano.
  Fact 0.27: Episode 111 overall originally aired on November 2, 2017.
  Fact 0.28: Episode 111 overall has production code 705.
  Fact 0.29: Episode 111 overall had 4.89 million U.S. viewers.
  Fact 0.30: Episode 112 overall is episode 6 of its season and is titled 'Vampires and Bloodsuckers'.
  Fact 0.31: Episode 112 overall was directed by Jann Turner.
  Fact 0.32: Episode 112 overall was written by Chris Van Dusen and Tia Napolitano.
  Fact 0.33: Episode 112 overall originally aired on November 9, 2017.
  Fact 0.34: Episode 112 overall has production code 706.
  Fact 0.35: Episode 112 overall had 5.00 million U.S. viewers.
  Fact 0.36: Episode 113 overall is episode 7 of its season and is titled 'Something Borrowed'.
  Fact 0.37: Episode 113 overall was directed by Sharat Raju.
  Fact 0.38: Episode 113 overall was written by Mark Fish.
  Fact 0.39: Episode 113 overall originally aired on November 16, 2017.
  Fact 0.40: Episode 113 overall has production code 707.
  Fact 0.41: Episode 113 overall had 4.97 million U.S. viewers.
  Fact 0.42: Episode 114 overall is episode 8 of its season and is titled 'Robin'.
  Fact 0.43: Episode 114 overall was directed by Daryn Okada.
  Fact 0.44: Episode 114 overall was written by Juan Carlos Fernandez.
  Fact 0.45: Episode 114 overall originally aired on January 18, 2018.
  Fact 0.46: Episode 114 overall has production code 709.
  Fact 0.47: Episode 114 overall had 5.17 million U.S. viewers.
  Fact 0.48: Episode 115 overall is episode 9 of its season and is titled 'Good People'.
  Fact 0.49: Episode 115 overall was directed by Nzingha Stewart.
  Fact 0.50: Episode 115 overall was written by Shonda Rhimes, Jess Brownell and Nicholas Nardini.
  Fact 0.51: Episode 115 overall originally aired on January 25, 2018.
  Fact 0.52: Episode 115 overall has production code 708.
  Fact 0.53: Episode 115 overall has U.S. viewership listed as TBD.
  Fact 0.54: Episode 116 overall is episode 10 of its season and is titled 'The People v. Olivia Pope'.
  Fact 0.55: Episode 116 overall has its director listed as TBA.
  Fact 0.56: Episode 116 overall has its writer listed as TBA.
  Fact 0.57: Episode 116 overall originally aired on February 1, 2018.
  Fact 0.58: Episode 116 overall has production code 710.
  Fact 0.59: Episode 116 overall has U.S. viewership listed as TBD.
  Fact 0.60: Episode 117 overall is episode 11 of its season and is titled 'Army of One'.
  Fact 0.61: Episode 117 overall has its director listed as TBA.
  Fact 0.62: Episode 117 overall has its writer listed as TBA.
  Fact 0.63: Episode 117 overall originally aired on February 8, 2018.
  Fact 0.64: Episode 117 overall has production code 711.
  Fact 0.65: Episode 117 overall has U.S. viewership listed as TBD.
  Fact 0.66: Episode 118 overall is episode 12 of its season.
  Fact 0.67: Episode 118 overall has its title, director, and writer listed as TBA.
  Fact 0.68: Episode 118 overall originally aired on March 1, 2018.
  Fact 0.69: Episode 118 overall has its production code listed as TBA.
  Fact 0.70: Episode 118 overall has U.S. viewership listed as TBD.
Sentence 1: 116 10 `` The People v. Olivia Pope '' TBA TBA February 1 , 2018 ( 2018 - 02 - 01 ) 710 TBD
  Fact 1.0: Episode 116 overall is episode 10 of its season and is titled 'The People v. Olivia Pope'.
  Fact 1.1: Episode 116 overall has its director listed as TBA.
  Fact 1.2: Episode 116 overall has its writer listed as TBA.
  Fact 1.3: Episode 116 overall originally aired on February 1, 2018.
  Fact 1.4: Episode 116 overall has production code 710.
  Fact 1.5: Episode 116 overall has U.S. viewership listed as TBD.

Retrieved items:
<item 1>
List of Scandal episodes
2017 ( 2017 - 10 - 26 ) 704 4.88 111 5 `` Adventures in Babysitting '' Oliver Bokelberg Serveriano Canales & Tia Napolitano November 2 , 2017 ( 2017 - 11 - 02 )
705 4.89 112 6 `` Vampires and Bloodsuckers '' Jann Turner Chris Van Dusen & Tia Napolitano November 9 , 2017 ( 2017 - 11 - 09 ) 706 5.00 113 7 `` Something
Borrowed '' Sharat Raju Mark Fish November 16 , 2017 ( 2017 - 11 - 16 ) 707 4.97 114 8 `` Robin '' Daryn Okada Juan Carlos Fernandez January 18 , 2018 ( 2018 -
01 - 18 ) 709 5.17 115 9 `` Good People '' Nzingha Stewart Shonda Rhimes , Jess Brownell & Nicholas Nardini January 25 ,
</item 1>

<item 2>
List of Scandal episodes
2017 ( 2017 - 04 - 20 ) 611 5.11 102 12 `` Mercy '' Nzingha Stewart Severiano Canales & Ameni Rozsa April 27 , 2017 ( 2017 - 04 - 27 ) 612 5.29 103 13 `` The
Box '' Steph Green Raamla Mohamed & Austin Guzman May 4 , 2017 ( 2017 - 05 - 04 ) 613 5.15 104 14 `` Head Games '' Zetna Fuentes Chris Van Dusen & Juan Carlos
Fernandez May 11 , 2017 ( 2017 - 05 - 11 ) 614 5.10 105 15 `` Tick Tock '' Salli Richardson - Whitfield Zahir McGhee & Michelle Lirtzman May 18 , 2017 ( 2017 -
05 - 18 ) 615 5.23 106 16 `` Transfer of Power '' Tony Goldwyn Matt Byrne & Mark Fish May 18 , 2017 ( 2017 - 05 - 18 ) 616 5.
</item 2>

<item 3>
List of Scandal episodes
2012 ( 2012 - 11 - 08 ) 205 6.06 13 6 `` Spies Like Us '' Bethany Rooney Chris Van Dusen November 15 , 2012 ( 2012 - 11 - 15 ) 206 6.02 14 7 `` Defiance '' Tom
Verica Peter Noah November 29 , 2012 ( 2012 - 11 - 29 ) 207 6.64 15 8 `` Happy Birthday , Mr. President '' Oliver Bokelberg Shonda Rhimes December 6 , 2012 (
2012 - 12 - 06 ) 208 7.39 16 9 `` Blown Away '' Jessica Yu Mark Wilding December 13 , 2012 ( 2012 - 12 - 13 ) 209 7.14 17 10 `` One for the Dog '' Steve Robin
Heather Mitchell January 10 , 2013 ( 2013 - 01 - 10 ) 210 8.37 18 11 `` A Criminal , a Whore ,
</item 3>

<item 4>
List of Scandal episodes
2017 ( 2017 - 03 - 16 ) 605 5.63 97 7 `` A Traitor Among Us '' Tom Verica Alison Schapker March 23 , 2017 ( 2017 - 03 - 23 ) 607 5.40 98 8 `` A Stomach for
Blood '' Oliver Bokelberg Severiano Canales March 30 , 2017 ( 2017 - 03 - 30 ) 608 6.57 99 9 `` Dead in the Water '' Nicole Rubio Michelle Lirtzman April 6 ,
2017 ( 2017 - 04 - 06 ) 609 5.10 100 10 `` The Decision '' Sharat Raju Johanna Lee April 13 , 2017 ( 2017 - 04 - 13 ) 610 5.35 101 11 `` Trojan Horse '' Jann
Turner Jess Brownell & Nicholas Nardini April 20 , 2017 ( 2017 - 04 - 20 ) 611 5.
</item 4>

<item 5>
List of Scandal episodes
2014 ( 2014 - 10 - 16 ) 404A 9.90 52 5 `` The Key '' Paul McCrane Chris Van Dusen October 23 , 2014 ( 2014 - 10 - 23 ) 404B 9.98 53 6 `` An Innocent Man ''
Jeannot Szwarc Zahir McGhee October 30 , 2014 ( 2014 - 10 - 30 ) 405 9.32 54 7 `` Baby Made a Mess '' Oliver Bokelberg Jenna Bans November 6 , 2014 ( 2014 - 11
- 06 ) 406 9.82 55 8 `` The Last Supper '' Julie Anne Robinson Allan Heinberg November 13 , 2014 ( 2014 - 11 - 13 ) 407 10.05 56 9 `` Where the Sun Do n't Shine
'' Tony Goldwyn Mark Wilding November 20 , 2014 ( 2014 - 11 - 20 ) 408 10.14 57 10 `` Run '' Tom Verica Shonda Rhimes January 29 ,
</item 5>

<item 6>
List of Scandal episodes
2013 ( 2013 - 03 - 21 ) 216 8.51 24 17 `` Snake in the Garden '' Ron Underwood Raamla Mohamed March 28 , 2013 ( 2013 - 03 - 28 ) 217 7.97 25 18 `` Molly , You
in Danger , Girl '' Tom Verica Chris Van Dusen April 4 , 2013 ( 2013 - 04 - 04 ) 218 8.04 26 19 `` Seven Fifty - Two '' Allison Liddi - Brown Mark Fish April 25
, 2013 ( 2013 - 04 - 25 ) 219 7.90 27 20 `` A Woman Scorned '' Tony Goldwyn Zahir McGhee May 2 , 2013 ( 2013 - 05 - 02 ) 220 8.07 28 21 `` Any Questions ? ''
</item 6>

<item 7>
List of Scandal episodes
2014 ( 2014 - 03 - 13 ) 313 8.22 43 14 `` Kiss Kiss Bang Bang '' Paul McCrane Mark Fish March 20 , 2014 ( 2014 - 03 - 20 ) 314 9.08 44 15 `` Mama Said Knock You
Out '' Tony Goldwyn Zahir McGhee March 27 , 2014 ( 2014 - 03 - 27 ) 315 9.01 45 16 `` The Fluffer '' Jeannot Szwarc Chris Van Dusen & Raamla Mohamed April 3 ,
2014 ( 2014 - 04 - 03 ) 316 9.13 46 17 `` Flesh and Blood '' Debbie Allen Severiano Canales & Miguel Nolla April 10 , 2014 ( 2014 - 04 - 10 ) 317 9.23 47 18 ``
The Price of Free and Fair Elections '' Tom Verica Shonda Rhimes & Mark Wilding April 17 , 2014 ( 2014 - 04 - 17 ) 318 10.
</item 7>

<item 8>
List of Scandal episodes
37 18 11 `` A Criminal , a Whore , an Idiot and a Liar '' Stephen Cragg Mark Fish January 17 , 2013 ( 2013 - 01 - 17 ) 211 7.93 19 12 `` Truth or Consequences
'' Jeannot Szwarc Peter Noah January 31 , 2013 ( 2013 - 01 - 31 ) 212 8.09 20 13 `` Nobody Likes Babies '' Tom Verica Mark Wilding February 7 , 2013 ( 2013 - 02
- 07 ) 213 8.14 21 14 `` Whiskey Tango Foxtrot '' Mark Tinker Matt Byrne February 14 , 2013 ( 2013 - 02 - 14 ) 214 8.02 22 15 `` Boom Goes the Dynamite '' Randy
Zisk Jenna Bans February 21 , 2013 ( 2013 - 02 - 21 ) 215 7.68 23 16 `` Top of the Hour '' Steve Robin Heather Mitchell March 21 , 2013 ( 2013 - 03 - 21 ) 216
8.
</item 8>

<item 9>
List of Scandal episodes
2015 ( 2015 - 03 - 12 ) 413 8.24 63 16 `` It 's Good to Be Kink '' Paul McCrane Matt Byrne March 19 , 2015 ( 2015 - 03 - 19 ) 414 7.79 64 17 `` Put a Ring on It
'' Regina King Chris Van Dusen March 26 , 2015 ( 2015 - 03 - 26 ) 415 8.06 65 18 `` Honor Thy Father '' Jeannot Szwarc Severiano Canales April 2 , 2015 ( 2015 -
04 - 02 ) 416 7.27 66 19 `` I 'm Just a Bill '' Debbie Allen Raamla Mohamed April 16 , 2015 ( 2015 - 04 - 16 ) 417 7.86 67 20 `` First Lady Sings the Blues ''
David Rodriguez Paul William Davies April 23 , 2015 ( 2015 - 04 - 23 ) 418 7.
</item 9>

<item 10>
List of Scandal episodes
2015 ( 2015 - 10 - 01 ) 502 9.12 72 `` Paris is Burning '' Jann Turner Matt Byrne October 8 , 2015 ( 2015 - 10 - 08 ) 503 8.76 73 `` Dog - Whistle Politics ''
Zetna Fuentes Mark Fish October 15 , 2015 ( 2015 - 10 - 15 ) 504 8.06 74 5 `` You Got Served '' Kevin Bray Zahir McGhee October 22 , 2015 ( 2015 - 10 - 22 ) 505
8.28 75 6 `` Get Out of Jail , Free '' Chandra Wilson Chris Van Dusen October 29 , 2015 ( 2015 - 10 - 29 ) 506 7.80 76 7 `` Even the Devil Deserves a Second
Chance '' Oliver Bokelberg Raamla Mohamed November 5 , 2015 ( 2015 - 11 - 05 ) 507 8.03 77 8 `` Rasputin '' John Terlesky Paul William Davies November 12 ,
</item 10>
=== END ITEM id=b72e8a3ed44997b6 ===

=== ITEM id=4e98686a8c3130d8 ===
Gold sentences and facts:
Sentence 0: Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium NRG Stadium Location Houston , Texas Operated 2006 -- present Conference tie - ins Big 12 , SEC Previous conference tie - ins Big East / C - USA / MWC ( alternating years ) ( 2006 -- 2009 ) Big Ten ( 2010 -- 2013 ) Payout US $ 3 million per team ( as of 2015 ) Preceded by Houston Bowl Sponsors Meineke Car Care ( 2011 -- 2012 ) AdvoCare ( 2014 -- 2016 ) Academy Sports + Outdoors ( 2017 - present ) Former names Texas Bowl ( 2006 -- 2010 ) Meineke Car Care Bowl of Texas ( 2011 -- 2012 ) Texas Bowl ( 2013 ) AdvoCare V100 Texas Bowl ( 2014 -- 2016 ) 2016 matchup Texas A&M vs. Kansas State ( Kansas State 33 -- 28 ) 2017 matchup Missouri vs Texas ( Texas 33 -- 16 )
  Fact 0.0: The Texas Bowl is also called Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium.
  Fact 0.1: The Texas Bowl is held at NRG Stadium.
  Fact 0.2: The Texas Bowl is located in Houston, Texas.
  Fact 0.3: The Texas Bowl has operated from 2006 to the present.
  Fact 0.4: The Texas Bowl's conference tie-ins are the Big 12 and the SEC.
  Fact 0.5: The Texas Bowl's previous conference tie-ins were the Big East, C-USA and MWC (alternating years) from 2006 to 2009.
  Fact 0.6: The Texas Bowl's previous conference tie-in was the Big Ten from 2010 to 2013.
  Fact 0.7: The Texas Bowl's payout is US $3 million per team as of 2015.
  Fact 0.8: The Texas Bowl was preceded by the Houston Bowl.
  Fact 0.9: Meineke Car Care sponsored the Texas Bowl from 2011 to 2012.
  Fact 0.10: AdvoCare sponsored the Texas Bowl from 2014 to 2016.
  Fact 0.11: Academy Sports + Outdoors has sponsored the Texas Bowl since 2017.
  Fact 0.12: A former name of the Texas Bowl was 'Texas Bowl' from 2006 to 2010.
  Fact 0.13: A former name of the Texas Bowl was 'Meineke Car Care Bowl of Texas' from 2011 to 2012.
  Fact 0.14: A former name of the Texas Bowl was 'Texas Bowl' in 2013.
  Fact 0.15: A former name of the Texas Bowl was 'AdvoCare V100 Texas Bowl' from 2014 to 2016.
  Fact 0.16: The 2016 Texas Bowl matchup was Texas A&M vs. Kansas State, which Kansas State won 33-28.
  Fact 0.17: The 2017 Texas Bowl matchup was Missouri vs Texas, which Texas won 33-16.

Retrieved items:
<item 1>
Texas Bowl
November 15 , 2017 . Jump up ^ Murphy , Austin ; Wetzel , Dan ( 15 November 2010 ) . `` Does It Matter ? '' . Sports Illustrated . p. 47 . External links ( edit
) Official Website Texas Bowl 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 NRG Stadium ( 2006 -- present ) Broadcasters Previous Bowls Bluebonnet Bowl
Houston Bowl NCAA Division I FBS bowl games College Football Playoff Championship Game Cotton Fiesta Orange Peach Rose Sugar Other bowl games Alamo Arizona
Armed Forces Bahamas Belk Birmingham Boca Raton Cactus Camellia Camping World Citrus Cure Dollar General Famous Idaho Potato Foster Farms Frisco Gasparilla
Hawaii Heart of Dallas Holiday Independence Las Vegas Liberty Military Music City New Mexico New Orleans Outback Pinstripe Quick Lane Sun TaxSlayer Texas All -
Star games East -- West Shrine Game NFLPA Collegiate Bowl Senior Bowl Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
Texas Bowl
In July 2006 , the NFL Network acquired TV rights and naming rights to the bowl . Texas Bowl introduction ( edit ) The Texas Bowl name and logo were officially
unveiled on August 10 , 2006 , at a press conference along with conference affiliations for the bowl spots . The Big 12 , Big East and Conference USA will be
affiliated with the game , as well as Texas Christian University of the Mountain West . The 2006 matchup featured teams from the Big 12 and Big East Conferences
. On December 3 , 2006 , Rutgers accepted an invitation to play Kansas State in the inaugural Texas Bowl . `` We 're ecstatic about having Rutgers , '' Texas
Bowl director David Brady said . `` This is a top - 15 team that was three yards away from a BCS game . We could n't be happier to have them here . '' 2010
marked the eleventh consecutive year a bowl game has played in Houston , and the 40th year overall with a bowl game there ( the Bluebonnet Bowl lasted 29 years
) .
</item 2>

<item 3>
Texas Bowl
It was also announced on December 30 , 2009 , that ESPN Plus would take over as part owner and operator of the game , while Lone Star Sports and Entertainment
will maintain a stake in the bowl , and would be carried on ESPN . Sponsors ( edit ) On April 12 , 2011 , ESPN announced Meineke Car Care signed a three - year
title sponsorship deal beginning in 2011 , changing name of the bowl to the Meineke Car Care Bowl of Texas . On February 12 , 2014 , it was announced that
AdvoCare will be the title sponsor for the bowl game . That sponsorship concluded after the 2016 game . On November 15 , 2017 , Academy Sports + Outdoors became
the new title sponsor of the bowl . Conference tie - ins ( edit ) On May 17 , 2007 , it was announced Conference USA would have a team in the 2007 Texas Bowl .
The Texas Bowl has a rotating commitment with the Big East Conference and Conference USA for 2006 -- 09 while the Big 12 Conference will have a team in all four
of those games .
</item 3>

<item 4>
Texas Bowl
Since 2017 , the game has been sponsored by Academy Sports + Outdoors . Contents ( hide ) 1 History 1.1 Replacing the Houston Bowl 1.2 Texas Bowl introduction
1.3 Sponsors 1.4 Conference tie - ins 1.5 Issues 2 Game results 3 MVPs 4 Most appearances 5 Appearances by conference 6 See also 7 References 8 External links
History ( edit ) Replacing the Houston Bowl ( edit ) Speculation surfaced questioning the long - term survival of the former Houston Bowl . The three - year
contract with EV1.net expired on December 31 , 2005 , leaving the bowl game without a title sponsor . A college football official told the Houston Chronicle
that the bowl was in danger of ceasing operations , as a result of the game losing its title sponsor and because the Houston Bowl still owed roughly $600,000 to
the Big 12 and Mountain West conferences following the 2005 game . However , the NCAA approved Lone Star Sports & Entertainment , a division of the Houston
Texans , who also play in Reliant Stadium , to take over game management .
</item 4>

<item 5>
Texas Bowl
Rank Conference Appearances Wins Losses Pct . Big 12 9 5 . 444 T2 SEC . 500 T2 Big Ten . 250 C - USA . 500 T5 ACC 0 1.000 T5 Big East 0 1.000 T5 Independents 0
1.000 T5 Mountain West 0 1.000 T5 MAC 0 . 000 Jump up ^ Rutgers ( 2006 ) appeared as a member of the Big East . Following the 2013 split of the original Big
East along football lines , the FBS schools reorganized as the new American Athletic Conference , which retains the charter of the original Big East . Jump up ^
Navy ( 2009 ) See also ( edit ) List of college bowl games List of Texas Bowl broadcasters References ( edit ) Jump up ^ `` College Bowl Game Payouts '' .
statisticbrain.com. 2015 . Jump up ^ Duarte , Joseph ( 18 April 2006 ) . `` Houston Bowl in jeopardy '' . Houston Chronicle . Retrieved 29 December 2015 . Jump
up ^ `` NFL Network gets bowl game in Houston '' .
</item 5>

<item 6>
Texas Bowl
In 2007 , TCU took the place of the Big 12 team when Kansas and Oklahoma were put into the BCS , and Houston , a `` home team , '' represented C - USA . The
conferences would receive $612,500 each as per the rules of the agreements as usually , the Big East ( or Big 12 ) would have received $750,000 for playing and
C - USA would have received a $500,000 stipend for their team playing . Issues ( edit ) According to Sports Illustrated , in 2008 the bowl required Western
Michigan University to purchase 11,000 tickets at full price in order to accept the invitation to play in the bowl . The university was only able to sell 548
tickets at that price , forcing it to accept a $462,535 loss , before travel expenses , to pay for the privilege of playing in the bowl . Game results ( edit )
Season Date Time ( CST ) Teams Attendance TV 2006 December 28 7 : 00 PM Rutgers 37 Kansas State 10 52,210 NFL Network 2007 December 28 7 : 00 PM TCU 20 Houston
13 62,
</item 6>

<item 7>
Texas Bowl
wikipedia.org/w/index.php?title=Texas_Bowl&oldid=817734724 '' Categories : College football bowls Texas Bowl American football in Texas Annual events in Texas
Annual sporting events in the United States Sports competitions in Texas Sports in Houston Recurring sporting events established in 2006 Talk Contents About
Wikipedia Français Edit links This page was last edited on 30 December 2017 , at 07 : 17 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Texas Bowl
Texas Bowl - wikipedia Texas Bowl Jump to : navigation , search `` AdvoCare V100 Bowl '' and `` AdvoCare Bowl '' redirect here . For the bowl game that AdvoCare
previously sponsored ( 2009 -- 13 ) , see Independence Bowl . Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium NRG Stadium Location Houston , Texas
Operated 2006 -- present Conference tie - ins Big 12 , SEC Previous conference tie - ins Big East / C - USA / MWC ( alternating years ) ( 2006 -- 2009 ) Big Ten
( 2010 -- 2013 ) Payout US $ 3 million per team ( as of 2015 ) Preceded by Houston Bowl Sponsors Meineke Car Care ( 2011 -- 2012 ) AdvoCare ( 2014 -- 2016 )
Academy Sports + Outdoors ( 2017 - present ) Former names Texas Bowl ( 2006 -- 2010 ) Meineke Car Care Bowl of Texas ( 2011 -- 2012 ) Texas Bowl ( 2013 )
AdvoCare V100 Texas Bowl ( 2014 -- 2016 ) 2016 matchup Texas A&M vs.
</item 8>

<item 9>
Texas Bowl
Kansas State ( Kansas State 33 -- 28 ) 2017 matchup Missouri vs Texas ( Texas 33 -- 16 ) The Texas Bowl , officially the Academy Sports + Outdoors Texas Bowl
for sponsorship purposes , is a post-season NCAA - sanctioned Division I FBS college football bowl game first held in 2006 in Houston , Texas . The bowl
replaced the defunct Houston Bowl , which played annually from 2000 to 2005 , and before that the Bluebonnet Bowl , the first bowl game in Houston , played from
1959 through 1987 . From 2011 to 2012 , the game was sponsored by Meineke Car Care and officially known as the Meineke Car Care Bowl of Texas . The game was
then officially the AdvoCare V100 Texas Bowl from 2014 to 2016 , with AdvoCare as the title sponsor . Since 2017 , the game has been sponsored by Academy Sports
+ Outdoors .
</item 9>

<item 10>
Texas Bowl
097 2008 December 30 7 : 00 PM Rice 38 Western Michigan 14 58,880 2009 December 31 2 : 30 PM Navy 35 Missouri 13 69,441 ESPN December 29 5 : 00 PM Illinois 38
Baylor 14 68,211 2011 December 31 11 : 00 AM Texas A&M 33 Northwestern 22 68,395 2012 December 28 8 : 00 PM Texas Tech 34 Minnesota 31 50,386 2013 December 27 5
: 00 PM Syracuse 21 Minnesota 17 32,327 2014 December 29 8 : 00 PM Arkansas 31 Texas 7 71,115 2015 December 29 8 : 00 PM LSU 56 Texas Tech 27 71,307 2016
December 28 8 : 00 PM Kansas State 33 Texas A&M 28 68,412 2017 December 27 8 : 00 PM Texas 33 Missouri 16 67,
</item 10>
=== END ITEM id=4e98686a8c3130d8 ===

=== ITEM id=e61d983cf26accac ===
Gold sentences and facts:
Sentence 0: Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout . He is the only Eagle Scout to have ascended to the U.S. Presidency .
  Fact 0.0: Ford was involved in the Boy Scouts of America.
  Fact 0.1: Ford earned the Boy Scouts of America program's highest rank, Eagle Scout.
  Fact 0.2: Ford is the only Eagle Scout to have ascended to the U.S. Presidency.

Retrieved items:
<item 1>
Gerald Ford
: Zable 1988 : Not presented 1989 : Ebert 1990 : Reagan 1991 : Gibson 1992 : Kemp 1993 : Alexander 1994 : Johnson 1995 : Mathias 1996 : Wooden 1997 : Payne 1998
: Dole 1999 : Richardson 2000 : Staubach 2001 : Cohen 2002 : Shriver 2003 : de Varona 2004 : Page 2005 : Ride 2006 : Kraft 2007 : Tagliabue 2008 : Glenn 2009 :
Albright 2010 : Mitchell 2011 : Dunwoody 2012 : Allen 2013 : Dungy 2014 : Mills 2015 : Jackson 2016 : Ueberroth 2017 : Brooke - Marciniak National Football
Foundation Gold Medal winners 1958 : Dwight D. Eisenhower 1959 : Douglas MacArthur 1960 : Herbert Hoover & Amos Alonzo Stagg 1961 : John F. Kennedy 1962 : Byron
`` Whizzer '' White 1963 : Roger Q. Blough 1964 : Donold B.
</item 1>

<item 2>
Gerald Ford
Capitol Complex , formerly House Annex 2 . Gerald R. Ford Freeway ( Nebraska ) Gerald R. Ford Freeway ( Michigan ) Gerald Ford Memorial Highway , I - 70 in
Eagle County , Colorado Gerald R. Ford International Airport in Grand Rapids , Michigan Gerald R. Ford Library in Ann Arbor , Michigan Gerald R. Ford Museum in
Grand Rapids , Michigan Gerald R. Ford School of Public Policy , University of Michigan Gerald R. Ford Institute of Public Policy , Albion College USS Gerald R.
Ford ( CVN - 78 ) Gerald R. Ford Middle School , Grand Rapids , Michigan President Gerald R. Ford Park in Alexandria , Virginia , located in the neighborhood
where Ford lived while serving as a Representative and Vice President President Ford Field Service Council , Boy Scouts of America The council where he was
awarded the rank of Eagle Scout . Serves 25 counties in Western and Northern Michigan with its headquarters located in Grand Rapids , Michigan . See also ( edit
) List of Freemasons List of Presidents of the United States List of Presidents of the United States by previous experience Presidents of the United States on
U.S. postage stamps References ( edit ) Jump up ^ Frum , David ( 2000 ) .
</item 2>

<item 3>
Gerald Ford
After the service , Ford was interred at his Presidential Museum in Grand Rapids , Michigan . Citizens lining outside of the Gerald R. Ford Presidential Museum
in Grand Rapids , Michigan during a public visitation of Ford . Scouting was so important to Ford that his family asked that Scouts participate in his funeral .
A few selected Scouts served as ushers inside the National Cathedral . About 400 Eagle Scouts were part of the funeral procession , where they formed an honor
guard as the casket went by in front of the museum . Ford selected the song to be played during his funeral procession at the U.S. Capitol . After his death in
December 2006 , the University of Michigan Marching Band played the school 's fight song for him one final time , for his last ride from the Gerald R. Ford
Airport in Grand Rapids , Michigan . The State of Michigan commissioned and submitted a statue of Ford to the National Statuary Hall Collection , replacing
Zachariah Chandler . It was unveiled on May 3 , 2011 in the Capitol Rotunda . On the proper right side is inscribed a quotation from a tribute by Thomas P.
</item 3>

<item 4>
Gerald Ford
Legacy of Honor : The Values and Influence of America 's Eagle Scouts . New York : St. Martin 's Press . pp. 12 -- 13 and 87 . ISBN 0 - 312 - 36653 - 1 .
Retrieved December 29 , 2006 . Jump up ^ `` Investigatory Records on Gerald Ford , Applicant for a Commission '' ( PDF ) . Gerald R. Ford Presidential Library .
December 30 , 1941 . Retrieved November 18 , 2010 . Jump up ^ Wertheimer , Linda ( December 27 , 2006 ) . `` Special Report : Former President Gerald Ford Dies
; Sought to Heal Nation Disillusioned by Watergate Scandal '' . National Public Radio . Retrieved April 26 , 2009 . Jump up ^ Perry , Will ( 1974 ) . `` No
Cheers From the Alumni '' . The Wolverines : A Story of Michigan Football ( PDF ) . Huntsville , Alabama : The Strode Publishers . pp. 150 -- 152 . ISBN 0 -
87397 - 055 - 1 .
</item 4>

<item 5>
Gerald Ford
I thank God for the gift of every sunrise and , even more , for all the years He has blessed me with Betty and the children ; with our extended family and the
friends of a lifetime . That includes countless Americans who , in recent months , have remembered me in their prayers . Your kindness touches me deeply . May
God bless you all and may God bless America . Ford 's age at the time of his death was 93 years and 165 days , making him the longest - lived U.S. President ,
his lifespan being 45 days longer than Ronald Reagan 's . He was the third - longest - lived Vice President , falling short only of John Nance Garner , 98 , and
Levi P. Morton , 96 . Ford also had the third - longest post-presidency ( 29 years and 11 months ) after Jimmy Carter ( 36 years , 9 months and counting ) and
Herbert Hoover ( 31 years and 7 months ) . Public image ( edit ) President George W. Bush with Ford and his wife Betty on April 23 , 2006 ; this is the last
known public photo of Gerald Ford .
</item 5>

<item 6>
President of the United States
Boy Scouts of America . Retrieved November 9 , 2017 . Jump up ^ Grier , Peter ( April 25 , 2011 ) . `` The ( not so ) secret history of the White House Easter
Egg Roll '' . The Christian Science Monitor . Archived from the original on July 30 , 2012 . Retrieved July 30 , 2012 . Jump up ^ Hesse , Monica ( November 21 ,
2007 ) . `` Turkey Pardons , The Stuffing of Historic Legend '' . The Washington Post . Retrieved May 14 , 2011 . Jump up ^ Gibbs , Nancy ( November 13 , 2008 )
. `` How Presidents Pass The Torch '' . Time . Retrieved May 6 , 2011 . Jump up ^ Dorning , Mike ( January 22 , 2009 ) . `` A note from Bush starts morning in
the Oval Office '' . Chicago Tribune . Archived from the original on December 28 , 2011 . Retrieved May 6 , 2011 .
</item 6>

<item 7>
President of the United States
Kennedy ( 1961 -- 1963 ) Lyndon B. Johnson ( 1963 -- 1969 ) Richard Nixon ( 1969 -- 1974 ) Gerald Ford ( 1974 -- 1977 ) Jimmy Carter ( 1977 -- 1981 ) Ronald
Reagan ( 1981 -- 1989 ) George H.W. Bush ( 1989 -- 1993 ) Bill Clinton ( 1993 -- 2001 ) George W. Bush ( 2001 -- 2009 ) Barack Obama ( 2009 -- 2017 ) Donald
Trump ( 2017 -- present ) Presidency timelines Wilson Harding Coolidge Hoover F.D. Roosevelt Truman Eisenhower Kennedy L.B. Johnson Nixon Ford Carter Reagan
G.H.W. Bush Clinton G.W. Bush Obama Trump Book Category Chief executives of the United States Federal President of the United States State governors ( current
list ) Alabama Alaska Arizona Arkansas California list Colorado Connecticut Delaware Florida Georgia Hawaii list Idaho Illinois list Indiana list Iowa list
Kansas Kentucky list Louisiana Maine list Maryland list Massachusetts Michigan list Minnesota list Mississippi Missouri Montana Nebraska list Nevada New
Hampshire list New Jersey list New Mexico list New York list North Carolina list North Dakota list Ohio Oklahoma list Oregon
</item 7>

<item 8>
President of the United States
`` George Washington 's views on political parties in America '' . The Washington Times . Retrieved July 1 , 2016 . Jump up ^ Chambers , William Nisbet ( 1963 )
. Political parties in a new Nation : the American experience , 1776 -- 1809 . New York : Oxford University Press . Jump up ^ `` Presidential and Vice
Presidential Salaries Exclusive of Perquisites '' . University of Michigan . Retrieved November 9 , 2017 . Jump up ^ Relative Value in US Dollars . Measuring
Worth . Retrieved May 30 , 2006 . Jump up ^ Dept. of Labor Inflation Calculator . Inflation Calculator . Retrieved August 10 , 2009 . Jump up ^ `` How much does
the U.S. president get paid ? '' . Howstuffworks . Retrieved July 24 , 2007 . ^ Jump up to : Salaries of Federal Officials : A Fact Sheet Archived April 26 ,
2013 , at the Wayback Machine ... United States Senate website . Retrieved August 6 , 2009 .
</item 8>

<item 9>
Gerald Ford
Ford was made a 33 ° Scottish Rite Mason on September 26 , 1962 . In April 1975 , Ford was elected by a unanimous vote Honorary Grand Master of the
International Supreme Council , Order of DeMolay , a position in which he served until January 1977 . Ford received the degrees of York Rite Masonry ( Chapter
and Council degrees ) in a special ceremony in the Oval Office on January 11 , 1977 , during his term as President of the United States . Honors ( edit ) Ford
honored on U.S Postage , issue of 2007 Ford received the Distinguished Eagle Scout Award in May 1970 , as well as the Silver Buffalo Award , from the Boy Scouts
of America . In 1974 , he also received the highest distinction of the Scout Association of Japan , the Golden Pheasant Award . In 1985 , he received the 1985
Old Tom Morris Award from the Golf Course Superintendents Association of America , GCSAA 's highest honor . In 1992 , the U.S. Navy Memorial Foundation awarded
Ford its Lone Sailor Award for his naval service and his subsequent government service .
</item 9>

<item 10>
Gerald Ford
That year his biological father , whom Ford described as a `` carefree , well - to - do man who did n't really give a damn about the hopes and dreams of his
firstborn son '' , approached Ford while he was waiting tables in a Grand Rapids restaurant . The two `` maintained a sporadic contact '' until Leslie King ,
Sr. 's death in 1941 . Ford said , `` My stepfather was a magnificent person and my mother equally wonderful . So I could n't have written a better prescription
for a superb family upbringing . '' Eagle Scout Gerald Ford ( circled in red ) in 1929 ; Michigan Governor Fred W. Green at far left , holding hat Ford as a
center on the University of Michigan football team , 1933 Ford was involved in the Boy Scouts of America , and earned that program 's highest rank , Eagle Scout
. He is the only Eagle Scout to have ascended to the U.S. Presidency . Ford attended Grand Rapids South High School , where he was a star athlete and captain of
the football team . In 1930 , he was selected to the All - City team of the Grand Rapids City League .
</item 10>
=== END ITEM id=e61d983cf26accac ===

=== ITEM id=48082513f1b6c197 ===
Gold sentences and facts:
Sentence 0: No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr - 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S. Tulsi Law Nominated 25 - Feb - 2014 24 - Feb - 2020 9 K. Parasaran Law Nominated 09 - Jun - 2012 28 - Jun - 2018 10 Rekha Art Nominated 27 - Apr - 2012 26 - Apr - 2018 11 Sachin Tendulkar Social service Nominated 27 - Apr - 2012 26 - Apr - 2018 12 Anu Aga Business Nominated 27 - Apr - 2012 26 - Apr - 2018
  Fact 0.0: Roopa Ganguly's field is Art and affiliation is Bharatiya Janata Party, with appointment on 04-Oct-2016 and retirement on 03-Oct-2022.
  Fact 0.1: Sambhaji Raje's field is Social work and affiliation is Bharatiya Janata Party, with appointment on 07-Jun-2016 and retirement on 03-May-2022.
  Fact 0.2: Suresh Gopi's field is Art and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.3: Subramanian Swamy's field is Economics and affiliation is Bharatiya Janata Party, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.4: Narendra Jadhav (No. 5) has field Economics and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.5: Mary Kom (No. 6) has field Sport and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.6: Swapan Dasgupta (No. 7) has field Journalism and affiliation Nominated, with appointment on 25-Apr-2016 and retirement on 24-Apr-2022.
  Fact 0.7: K.T.S. Tulsi (No. 8) has field Law and affiliation Nominated, with appointment on 25-Feb-2014 and retirement on 24-Feb-2020.
  Fact 0.8: K. Parasaran (No. 9) has field Law and affiliation Nominated, with appointment on 09-Jun-2012 and retirement on 28-Jun-2018.
  Fact 0.9: Rekha (No. 10) has field Art and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.10: Sachin Tendulkar (No. 11) has field Social service and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.
  Fact 0.11: Anu Aga (No. 12) has field Business and affiliation Nominated, with appointment on 27-Apr-2012 and retirement on 26-Apr-2018.

Retrieved items:
<item 1>
List of nominated members of Rajya Sabha
Dutta 1993 -- 1999 85 -- Maulana Habibur Rahman Nomani 1993 -- 1999 86 Nirmala Deshpande 1997 -- 1999 and 2004 -- 2010 87 Shabana Azmi 1997 -- 2003 88 -- P.
Selvie Das 1997 -- 2003 89 -- Kartar Singh Duggal 1997 -- 2003 90 Kuldip Nayyar 1997 -- 2003 91 -- Raja Ramanna 1997 -- 2003 92 C.
</item 1>

<item 2>
List of nominated members of Rajya Sabha
Daphtary 1972 -- 1978 47 -- Habib Tanvir 1972 -- 1978 48 -- Krishna Kriplani 1974 -- 1980 49 Lokesh Chandra 1974 -- 1986 50 -- Scato Swu 1974 -- 1986 51 -- B.N.
Banerjee 1976 -- 1982 52 -- Malcolm Adiseshiah 1978 -- 1984 53 -- Fathema Ismail 1978 -- 1984 54 -- Pandurang Dharmaji Jadhav 1978 -- 1984 55 -- Bhagwati Charan
Varma 1978 -- 1984 56 Nargis Dutt 1980 -- 1986 57 Khushwant Singh 1980 -- 1986 58 -- Asima Chatterjee 1982 -- 1990 59 Sivaji Ganesan 1982 -- 1986 60 --
Hayatullah Ansari 1982 -- 1988 61 -- Madan Bhatia 1982 -- 1994 62 -- V.N.
</item 2>

<item 3>
List of nominated members of Rajya Sabha
Tiwari 1982 -- 1984 63 -- Ghulam Rasool Kar 1984 -- 1987 64 -- Tindivanam K. Ramamurthy 1984 -- 1990 65 -- H.L. Kapur 1985 -- 1988 66 Salim Ali 1985 -- 1988 67
-- Purushottam Kakodkar 1985 -- 1991 68 Ela Bhatt 1986 -- 1988 69 -- Amrita Pritam 1986 -- 1992 70 M.F. Husain 1986 -- 1992 71 R.K.
</item 3>

<item 4>
List of nominated members of Rajya Sabha
Setalvad 1966 -- 1972 35 G. Sankara Kurup 1968 -- 1972 36 Joachim Alva 1968 -- 1974 37 S. Nurul Hasan 1968 -- 1974 38 -- K. Ramiah 1968 -- 1974 39 -- Ganga
Sharan Sinha 1968 -- 1974 40 -- Maragatham Chandrasekhar 1970 -- 1988 41 -- Umashankar Joshi 1970 -- 1976 42 -- Rasheeduddin Khan 1970 -- 1982 43 -- Vidya
Prakash Dutt 1971 -- 1980 44 -- Abu Abraham 1972 -- 1978 45 -- Pramathanath Bishi 1972 -- 1978 46 -- C.K.
</item 4>

<item 5>
List of nominated members of Rajya Sabha
No . Name Field Affiliation Date of Appointment Date of Retirement Roopa Ganguly Art Bharatiya Janata Party 04 - Oct - 2016 03 - Oct - 2022 Sambhaji Raje Social
work Bharatiya Janata Party 07 - Jun - 2016 03 - May - 2022 Suresh Gopi Art Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 Subramanian Swamy Economics
Bharatiya Janata Party 25 - Apr - 2016 24 - Apr - 2022 5 Narendra Jadhav Economics Nominated 25 - Apr - 2016 24 - Apr - 2022 6 Mary Kom Sport Nominated 25 - Apr
- 2016 24 - Apr - 2022 7 Swapan Dasgupta Journalism Nominated 25 - Apr - 2016 24 - Apr - 2022 8 K.T.S.
</item 5>

<item 6>
List of nominated members of Rajya Sabha
Kasturirangan 2003 -- 2009 102 -- Narayan Singh Manaklao 2003 -- 2009 103 -- Vidya Niwas Misra 2003 -- 2009 104 -- Chandan Mitra 2003 -- 2009 105 Dara Singh
2003 -- 2009 106 Kapila Vatsyayan 2006 -- 2012 107 Ram Jethmalani 2006 -- 2009 108 -- Shobhana Bhartia 2006 -- 2012 109 Shyam Benegal 2006 -- 2012 110 M.S.
Swaminathan 2007 -- 2013 111 -- C. Rangarajan 2008 -- 2010 112 -- H.K.
</item 6>

<item 7>
List of nominated members of Rajya Sabha
Narayana Reddy 1997 -- 2003 93 Mrinal Sen 1997 -- 2003 94 -- Chaudhary Harmohan Singh Yadav 1997 -- 2003 95 -- Nanaji Deshmukh 1999 -- 2005 96 Lata Mangeshkar
1999 -- 2005 97 -- Fali Nariman 1999 -- 2005 98 -- Cho Ramaswamy 1999 -- 2005 99 Hema Malini 2003 -- 2009 100 -- Bimal Jalan 2003 -- 2009 101 K.
</item 7>

<item 8>
List of nominated members of Rajya Sabha
Krishnamachari 1961 -- 1967 24 -- R.R. Diwakar 1962 -- 1968 25 -- Gopal Singh 1962 -- 1968 26 -- M. Ajmal Khan 1964 -- 1972 27 -- Satyavrata Siddhantalankar
1964 -- 1968 28 Shakuntala Paranjpye 1964 -- 1970 29 -- Badri Nath Prasad 1964 -- 1970 30 -- G. Ramachandran 1964 -- 1970 31 -- M.N. Kaul 1964 -- 1972 32 --
Harivanshrai Bachchan 1966 -- 1972 33 -- Dhananjay Ramchandra Gadgil 1966 -- 1972 34 -- M.C.
</item 8>

<item 9>
List of nominated members of Rajya Sabha
Tulsi 2014 - 2020 125 Subramanian Swamy 2016 - 2022 126 MC Mary Kom 2016 - 2022 127 Narendra Jadhav 2016 - 2022 129 Suresh Gopi 2016 - 2022 130 -- Swapan
Dasgupta 2016 - 2022 131 -- Sambhaji Raje 2016 - 2022 132 -- Roopa Ganguly 2016 - 2022 References ( edit ) Nominated Members since 1952 Narender Kumar Current
list of Nominated Members Specific Jump up ^ `` Statewise List '' . 164.100. 47.5 . Retrieved 12 June 2016 . Jump up ^
http://164.100.47.5/Newmembers/partymemberlist.aspx ^ Jump up to : `` Statewise Retirement '' . 164.100. 47.5 . Retrieved 12 June 2016 .
</item 9>

<item 10>
List of nominated members of Rajya Sabha
Sr. No . Image Name Tenure -- Alladi Krishnaswamy Iyer 1952 -- 1953 Satyendranath Bose 1952 -- 1960 Prithviraj Kapoor 1952 -- 1960 -- Jagadisan Mohandas
Kumarappa 1952 -- 1954 5 -- Kalidas Nag 1952 -- 1954 6 Rukmini Devi Arundale 1952 -- 1962 7 -- N.R. Malkani 1952 -- 1962 8 -- Sahib Singh Sokhey 1952 -- 1956 9
-- Zakir Hussain 1952 -- 1962 10 -- Maithili Sharan Gupt 1952 -- 1964 11 -- Kakasaheb Kalelkar 1952 -- 1964 12 -- Radha Kumud Mukherjee 1952 -- 1958 13 P.V.
</item 10>
=== END ITEM id=48082513f1b6c197 ===
