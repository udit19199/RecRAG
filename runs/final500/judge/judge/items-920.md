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

This file has 6 items.

=== ITEM id=730d4b3eee88043d ===
Gold sentences and facts:
Sentence 0: Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait * Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org
  Fact 0.0: The Organization of the Petroleum Exporting Countries (OPEC) has its headquarters in Vienna, Austria.
  Fact 0.1: OPEC's official language is English.
  Fact 0.2: OPEC is an international cartel.
  Fact 0.3: OPEC has 14 member states as of May 2017.
  Fact 0.4: The OPEC member states listed are Algeria, Angola, Ecuador, Equatorial Guinea, Gabon, Iran, Iraq, Kuwait, Libya, Nigeria, Qatar, Saudi Arabia, the United Arab Emirates, and Venezuela.
  Fact 0.5: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela are marked with an asterisk, which the table defines as founding member.
  Fact 0.6: The OPEC leader is Secretary General Mohammed Barkindo.
  Fact 0.7: OPEC was established in Baghdad, Iraq.
  Fact 0.8: The OPEC statute is dated September 1960 and was in effect from January 1961.
  Fact 0.9: OPEC's prices are indexed as US dollars per barrel (US$/bbl).
  Fact 0.10: OPEC's website is OPEC.org.
Sentence 1: Organization of the Petroleum Exporting Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria . As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies .
  Fact 1.0: OPEC is an intergovernmental organization of 14 nations as of February 2018.
  Fact 1.1: OPEC is pronounced OH-pek (IPA: ˈoʊpɛk), or OPEP in several other languages.
  Fact 1.2: OPEC was founded in 1960 in Baghdad by its first five members: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.
  Fact 1.3: OPEC has been headquartered in Vienna, Austria since 1965.
  Fact 1.4: As of 2016, the 14 OPEC countries accounted for an estimated 44 percent of global oil production.
  Fact 1.5: As of 2016, the 14 OPEC countries held an estimated 73 percent of the world's 'proven' oil reserves.
  Fact 1.6: OPEC has a major influence on global oil prices.
  Fact 1.7: Global oil prices were previously determined by American-dominated multinational oil companies.
Sentence 2: Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa 1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 -- 1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa 1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,000 Iraq Middle East 1960 -- 37,202,572 437,072 4,451,516 143,069,000,000 Kuwait Middle East 1960 -- 4,052,584 17,820 2,923,825 101,500,000,000 Libya North Africa 1962 -- 6,293,253 1,759,540 384,686 48,363,000,000 Nigeria West Africa 1971 -- 185,989,640 923,768 1,999,885 37,070,000,000 Qatar Middle East 1961 -- 2,569,804 11,437 1,522,902 25,244,000,000 Saudi Arabia Middle East 1960 -- 32,275,687 2,149,690 10,460,710 266,578,000,000 United Arab Emirates Middle East 1967 -- 9,269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695 35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 %
  Fact 2.0: Algeria (North Africa): membership from 1969; population 40,606,052 (2016 est.); area 2,381,740 km²; oil production 1,348,361 bbl/day (2016); proven reserves 12,200,000,000 bbl (2016).
  Fact 2.1: Angola (Southern Africa): membership from 2007; population 28,813,463; area 1,246,700 km²; oil production 1,769,615 bbl/day; proven reserves 8,423,000,000 bbl.
  Fact 2.2: Ecuador (South America): membership 1973 to 1992 and from 2007; population 16,385,068; area 283,560 km²; oil production 548,421 bbl/day; proven reserves 8,273,000,000 bbl.
  Fact 2.3: Equatorial Guinea (Central Africa): membership from 2017; population 1,221,490; area 28,051 km²; oil production 227,000 bbl/day; proven reserves 1,100,000,000 bbl.
  Fact 2.4: Gabon (Central Africa): membership 1975 to 1995 and from 2016; population 1,979,786; area 267,667 km²; oil production 210,820 bbl/day; proven reserves 2,000,000,000 bbl.
  Fact 2.5: Iran (Middle East): membership from 1960; population 80,277,428; area 1,648,000 km²; oil production 3,990,956 bbl/day; proven reserves 157,530,000,000 bbl.
  Fact 2.6: Iraq (Middle East): membership from 1960; population 37,202,572; area 437,072 km²; oil production 4,451,516 bbl/day; proven reserves 143,069,000,000 bbl.
  Fact 2.7: Kuwait (Middle East): membership from 1960; population 4,052,584; area 17,820 km²; oil production 2,923,825 bbl/day; proven reserves 101,500,000,000 bbl.
  Fact 2.8: Libya (North Africa): membership from 1962; population 6,293,253; area 1,759,540 km²; oil production 384,686 bbl/day; proven reserves 48,363,000,000 bbl.
  Fact 2.9: Nigeria (West Africa): membership from 1971; population 185,989,640; area 923,768 km²; oil production 1,999,885 bbl/day; proven reserves 37,070,000,000 bbl.
  Fact 2.10: Qatar (Middle East): membership from 1961; population 2,569,804; area 11,437 km²; oil production 1,522,902 bbl/day; proven reserves 25,244,000,000 bbl.
  Fact 2.11: Saudi Arabia (Middle East): membership from 1960; population 32,275,687; area 2,149,690 km²; oil production 10,460,710 bbl/day; proven reserves 266,578,000,000 bbl.
  Fact 2.12: United Arab Emirates (Middle East): membership from 1967; population 9,269,612; area 83,600 km²; oil production 3,106,077 bbl/day; proven reserves 97,800,000,000 bbl.
  Fact 2.13: Venezuela (South America): membership from 1960; population 31,568,179; area 912,050 km²; oil production 2,276,967 bbl/day; proven reserves 299,953,000,000 bbl.
  Fact 2.14: OPEC total: population 478,498,000; area 12,150,695 km²; oil production 35,221,740 bbl/day; proven reserves 1,209,103,000,000 bbl.
  Fact 2.15: World total: population 7,618,511,000; area 510,072,000 km²; oil production 80,622,287 bbl/day; proven reserves 1,650,585,000,000 bbl.
  Fact 2.16: OPEC percent of the world: population 6%; area 2%; oil production 44%; proven reserves 73%.

Retrieved items:
<item 1>
OPEC
A statement released by OPEC on 10 September 2008 confirmed Indonesia 's withdrawal , noting that OPEC `` regretfully accepted the wish of Indonesia to suspend
its full membership in the organization , and recorded its hope that the country would be in a position to rejoin the organization in the not - too - distant
future . '' 2008 production dispute ( edit ) Countries by net oil exports ( 2008 ) The differing economic needs of OPEC member states often affect the internal
debates behind OPEC production quotas . Poorer members have pushed for production cuts from fellow members , to increase the price of oil and thus their own
revenues . These proposals conflict with Saudi Arabia 's stated long - term strategy of being a partner with the world 's economic powers to ensure a steady
flow of oil that would support economic expansion . Part of the basis for this policy is the Saudi concern that overly expensive oil or unreliable supply will
drive industrial nations to conserve energy and develop alternative fuels , curtailing the worldwide demand for oil and eventually leaving unneeded barrels in
the ground . To this point , Saudi Oil Minister Yamani famously remarked in 1973 : `` The Stone Age did n't end because we ran out of stones . ''
</item 1>

<item 2>
OPEC
OPEC - wikipedia OPEC Jump to : navigation , search Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official
language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait *
Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq
Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org Organization of the Petroleum Exporting
Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded
in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria .
</item 2>

<item 3>
OPEC
Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa
1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 --
1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa
1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,
</item 3>

<item 4>
OPEC
In May 2008 , Indonesia announced that it would leave OPEC when its membership expired at the end of that year , having become a net importer of oil and being
unable to meet its production quota . It rejoined the organization in January 2016 , but announced another `` temporary suspension '' of its membership at year
- end when OPEC requested a 5 percent production cut . Some commentators consider that the United States was a de facto member of OPEC during its formal
occupation of Iraq , due to its leadership of the Coalition Provisional Authority in 2003 -- 2004 . But this is not borne out by the minutes of OPEC meetings ,
as no US representative attended in an official capacity . Observers ( edit ) Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia ,
and other oil - exporting nations have attended many OPEC meetings as observers . This arrangement serves as an informal mechanism for coordinating policies .
</item 4>

<item 5>
OPEC
Jump up ^ The UAE was founded in December 1971 . Its OPEC membership originated with the Emirate of Abu Dhabi . Lapsed members ( edit ) Country Region
Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Indonesia Southeast Asia 1962 -- 2008
, Jan - Nov 2016 261,115,456 1,904,569 833,667 3,692,500,000 For countries that export petroleum at relatively low volume , their limited negotiating power as
OPEC members would not necessarily justify the burdens imposed by OPEC production quotas and membership costs . Ecuador withdrew from OPEC in December 1992 ,
because it was unwilling to pay the annual US $2 million membership fee and felt that it needed to produce more oil than it was allowed under its OPEC quota at
the time , although it rejoined in October 2007 . Similar concerns prompted Gabon to suspend membership in January 1995 ; it rejoined in July 2016 .
</item 5>

<item 6>
OPEC
269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695
35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 % ^ Jump up to : One petroleum
barrel ( bbl ) is approximately 42 U.S. gallons , or 159 liters , or 0.159 m , varying slightly with temperature . To put the production numbers in context , a
supertanker typically holds 2,000,000 barrels ( 320,000 m ) , and the world 's current production rate would take approximately 56 years to exhaust the world 's
current proven reserves . ^ Jump up to : The five founding members attended the first OPEC conference in September 1960 .
</item 6>

<item 7>
OPEC
During 1961 -- 1975 , the five founding nations were joined by Qatar ( 1961 ) , Indonesia ( 1962 -- 2008 , rejoined 2014 - 2016 ) , Libya ( 1962 ) , United Arab
Emirates ( originally just the Emirate of Abu Dhabi , 1967 ) , Algeria ( 1969 ) , Nigeria ( 1971 ) , Ecuador ( 1973 -- 1992 , rejoined 2007 ) , and Gabon ( 1975
-- 1994 , rejoined 2016 ) . By the early 1970s , OPEC 's membership accounted for more than half of worldwide oil production . Indicating that OPEC is not
averse to further expansion , Mohammed Barkindo , OPEC 's Acting Secretary General in 2006 , urged his African neighbors Angola and Sudan to join , and Angola
did in 2007 , followed by Equatorial Guinea in 2017 . Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia , and other oil - exporting
nations have attended many OPEC meetings as observers , as an informal mechanism for coordinating policies .
</item 7>

<item 8>
OPEC
9 2003 -- 2011 volatility 3.10 2008 production dispute 3.11 2014 -- 2017 oil glut 3.12 2017 - 2018 Production cut 4 See also 5 References 6 External links
Membership ( edit ) Current member countries ( edit ) As of May 2017 , OPEC has 14 member countries : six in the Middle East ( Western Asia ) , six in Africa ,
and two in South America . According to the U.S. Energy Information Administration ( EIA ) , OPEC 's combined rate of oil production ( including gas condensate
) represented 44 percent of the world 's total in 2016 , and OPEC accounted for 73 percent of the world 's `` proven '' oil reserves , including 48 percent from
just the six Middle Eastern members : Approval of a new member country requires agreement by three - quarters of OPEC 's existing members , including all five
of the founders . In October 2015 , Sudan formally submitted an application to join , but it is not yet a member . Country Region Membership Years Population (
2016 est . )
</item 8>

<item 9>
OPEC
As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving
OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies . OPEC 's stated mission is ``
to coordinate and unify the petroleum policies of its member countries and ensure the stabilization of oil markets , in order to secure an efficient , economic
and regular supply of petroleum to consumers , a steady income to producers , and a fair return on capital for those investing in the petroleum industry . ''
The organization is also a significant provider of information about the international oil market . As of May 2017 , OPEC 's members are Algeria , Angola ,
Ecuador , Equatorial Guinea , Gabon , Iran , Iraq , Kuwait , Libya , Nigeria , Qatar , Saudi Arabia ( the de facto leader ) , United Arab Emirates , and
Venezuela , while Indonesia is a former member . Two - thirds of OPEC 's oil production and reserves are in its six Middle Eastern countries that surround the
oil - rich Persian Gulf .
</item 9>

<item 10>
OPEC
However , their influence on international trade is periodically challenged by the expansion of non-OPEC energy sources , and by the recurring temptation for
individual OPEC countries to exceed production ceilings and pursue conflicting self - interests . Contents ( hide ) 1 Membership 1.1 Current member countries
1.2 Lapsed members 1.3 Observers 1.4 Leadership and decision - making 1.5 International cartel 1.6 Conflicts 2 Market information 2.1 Publications and research
2.2 Crude oil benchmarks 2.3 Spare capacity 3 History and impact 3.1 Post-WWII situation 3.2 1959 -- 1960 anger from exporting countries 3.3 1960 -- 1975
founding and expansion 3.4 1973 -- 1974 oil embargo 3.5 1975 -- 1980 Special Fund , now OFID 3.6 1975 hostage siege 3.7 1979 -- 1980 oil crisis and 1980s oil
glut 3.8 1990 -- 2003 ample supply and modest disruptions 3.9 2003 -- 2011 volatility 3.
</item 10>
=== END ITEM id=730d4b3eee88043d ===

=== ITEM id=162142f0ed611dab ===
Gold sentences and facts:
Sentence 0: The season is scheduled to be released on March 8 , 2018 .
  Fact 0.0: The season is scheduled to be released on March 8, 2018.

Retrieved items:
<item 1>
Jessica Jones (season 2)
They are joined by Leah Gibson and J.R. Ramirez . The second season was ordered in January 2016 , with filming beginning in April 2017 , back - to - back with
the miniseries Marvel 's The Defenders . Filming concluded in September 2017 . The season is scheduled to be released on March 8 , 2018 . Contents ( hide ) 1
Cast 1.1 Main 1.2 Notable guests 2 Production 2.1 Development 2.2 Writing 2.3 Casting 2.4 Filming 3 Release 3.1 Marketing 4 References 5 External links Cast (
edit ) Further information : List of Jessica Jones characters Main ( edit ) Krysten Ritter as Jessica Jones Rachael Taylor as Patricia `` Trish '' Walker Carrie
- Anne Moss as Jeri Hogarth Eka Darville as Malcolm Ducasse Leah Gibson as Ingrid J.R.
</item 1>

<item 2>
Jessica Jones (season 2)
`` JESSICA JONES Renewed For Second Season '' . Newsarama . Archived from the original on January 18 , 2016 . Retrieved January 18 , 2016 . ^ Jump up to : Beard
, Jim ( February 3 , 2016 ) . `` Rachael Taylor Deals with Trish Walker 's Inner Demons in ' Marvel 's Jessica Jones ' '' . Marvel.com . Archived from the
original on February 5 , 2016 . Retrieved February 5 , 2016 . ^ Jump up to : Beard , Jim ( February 5 , 2016 ) . `` Carrie - Anne Moss Pays for Hogarth 's Sins
in ' Marvel 's Jessica Jones ' '' . Marvel.com . Archived from the original on February 6 , 2016 . Retrieved February 6 , 2016 . ^ Jump up to : Perry , Spencer
( August 17 , 2017 ) . `` Jessica Jones Season 2 Set Photos Tease How Kilgrave Returns '' . ComingSoon.net .
</item 2>

<item 3>
Jessica Jones (season 2)
Jessica Jones ( season 2 ) - wikipedia Jessica Jones ( season 2 ) Jump to : navigation , search Jessica Jones ( season 2 ) Promotional poster Starring Krysten
Ritter Rachael Taylor Carrie - Anne Moss Eka Darville Leah Gibson J.R. Ramirez Country of origin United States Release Original network Netflix Season
chronology ← Previous Season 1 List of Jessica Jones episodes The second season of the American web television series Jessica Jones , which is based on the
Marvel Comics character of the same name , follows Jones as she takes on a new case after the events surrounding her encounter with Kilgrave . It is set in the
Marvel Cinematic Universe ( MCU ) , sharing continuity with the films and other television series of the franchise . The season will be produced by Marvel
Television in association with ABC Studios and Tall Girls Productions , with Melissa Rosenberg serving as showrunner . Krysten Ritter stars as Jones , with
Rachael Taylor , Carrie - Anne Moss , and Eka Darville also returning from the first season . They are joined by Leah Gibson and J.R. Ramirez .
</item 3>

<item 4>
Jessica Jones (season 2)
On January 17 , 2016 , Netflix ordered a second season of 13 episodes . Raelle Tucker joined the season as an executive producer and writer , replacing Liz
Friedman from the first season , who departed the series to work on the pilot for the ABC series , Conviction . Writing ( edit ) Rosenberg and the season 's
writers were halfway through the writing process by August 2016 , with the scripts completed by the end of October 2016 . With The Defenders releasing before
the season , Rosenberg used the miniseries as an opportunity to help `` set up '' elements for the season , working with The Defenders executive producers and
writers Doug Petrie and Marco Ramirez to do so . Rosenberg wanted to `` continue with ( the Jessica Jones ) character '' in the season , saying , `` She 's a
very damaged character , her damage goes beyond ( David Tennant 's ) Kilgrave . There 's a lot to mine from in her backstory and in her present day situation ''
.
</item 4>

<item 5>
Jessica Jones (season 2)
Retrieved September 7 , 2016 . External links ( edit ) Marvel 's Jessica Jones on IMDb Marvel 's Jessica Jones at TV.com Jessica Jones Brian Michael Bendis
Michael Gaydos Supporting characters Luke Cage Carol Danvers Jessica Drew Ben Urich Enemies Man Mountain Marko Purple Man Publications Alias The Pulse Jessica
Jones In other media Jessica Jones season 1 characters Related articles Daily Bugle New Avengers Marvel Cinematic Universe Feature films Phase One Iron Man The
Incredible Hulk Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The
Winter Soldier Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2
Spider - Man : Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television
series ABC Agents of S.H.I.E.L.D.
</item 5>

<item 6>
Jessica Jones (season 2)
Retrieved August 14 , 2017 . Jump up ^ Schwartz , Terri ( January 7 , 2015 ) . `` Netflix announces ' Daredevil , ' ' Bloodline , ' ' Kimmy Schmidt ' premiere
dates '' . Zap2It . Archived from the original on January 10 , 2015 . Retrieved January 10 , 2015 . Jump up ^ Goldman , Eric ( July 28 , 2015 ) . `` Netflix On
Marvel Series Release Plan And If Punisher Could Get A Spinoff '' . IGN . Archived from the original on July 28 , 2015 . Retrieved July 28 , 2015 . Jump up ^
Weintraub , Steve ( October 12 , 2015 ) . `` Krysten Ritter & Showrunner Melissa Rosenberg Talk ' Jessica Jones ' at NYCC '' . Collider . Archived from the
original on October 14 , 2015 . Retrieved October 15 , 2015 . Jump up ^ Ryan , Maureen ( November 20 , 2015 ) .
</item 6>

<item 7>
Jessica Jones (season 2)
This was a goal that `` Marvel was completely on board with '' , and given the demand of many talented female directors at the time , the series ' producers
looked to book only female directors first , and approach male directors later in the pre-production phase if needed . Another member of the production
suggested that the series book only female directors for the season , which Rosenberg `` had n't contemplated ( as a ) concept prior to that conversation '' .
She quickly made that the goal of the production , and in October 2016 , Rosenberg confirmed that all 13 episodes of the season would be directed by women .
Release ( edit ) The second season of Jessica Jones is scheduled to be released on March 8 , 2018 , on the streaming service Netflix worldwide , in Ultra HD 4K
and high dynamic range . Marketing ( edit ) In December 2017 , a teaser trailer for the season was released , along with announcing the season release date .
References ( edit ) ^ Jump up to : Arrant , Chris ( January 17 , 2016 ) . `` JESSICA JONES Renewed For Second Season '' . Newsarama .
</item 7>

<item 8>
Jessica Jones (season 2)
`` Netflix hit ' Jessica Jones ' films at Long Island Aquarium '' . Northforker . Archived from the original on June 12 , 2017 . Retrieved June 12 , 2017 . Jump
up ^ Ridgely , Charlie ( September 15 , 2017 ) . `` Filming Wraps on ' Jessica Jones ' Season 2 '' . Comicbook.com . Archived from the original on September 15
, 2017 . Retrieved September 15 , 2017 . ^ Jump up to : Ramos , Dino - Ray ( December 9 , 2017 ) . `` ' Marvel 's Jessica Jones ' : Season 2 Premiere Date Set
For Netflix Superhero Series '' . Deadline.com . Archived from the original on December 10 , 2017 . Retrieved December 10 , 2017 . Jump up ^ `` Marvel 's
Jessica Jones '' . Netflix . Archived from the original on September 7 , 2016 . Retrieved September 7 , 2016 .
</item 8>

<item 9>
Jessica Jones (season 2)
People do n't just heal '' . She added that Jones ' killing of Kilgrave at the end of the first season was `` a life changing experience '' and something that
would affect the character going forward . Rosenberg also stated that the season would be `` about digging deeper into this chaos and peeling back those layers
( of Jones ' life ) , just going to the core of her being '' after the first season focused on Jones ' trauma and facing her abuser . Casting ( edit ) After the
season was ordered , several main cast members revealed that they would return for the second season , including Ritter as Jessica Jones , Rachael Taylor as
Patricia `` Trish '' Walker , and Carrie - Anne Moss as Jeri Hogarth . Eka Darville also reprises his role as Malcolm Ducasse . In March 2017 , J.R. Ramirez was
cast as Oscar , which was revealed in July after the airing of his character 's death on Power . By that month , Leah Gibson had also joined the cast , in the
role of Ingrid .
</item 9>

<item 10>
Jessica Jones (season 2)
On Jones becoming famous after her heroics in the first season , Ritter said , `` She keeps her circle small because she does n't want people in her life , so
there 's no textbook on how to deal with new popularity or new eyes on you . '' Speaking on the social issues she hoped to tackle in the season , after covering
`` issues of choice , interracial relationships , domestic violence , ( and ) issues of consent '' while also exploring `` feminism and being a woman in this
world '' in the first , Rosenberg said , `` I 'm not quite sure yet what the social issues are that we 're dealing with ( in season 2 ) . We 're just trying to
find some resonance for ( Jessica Jones ) and a new place to push her , to give Krysten something new to play and really push the boundaries of the character .
'' After much of the first season was taken from the Alias comic book , Rosenberg wanted to continue that trend with the second season , but acknowledged that
`` the MCU is very different from the comics in terms of its mythology . In the books there were things building towards Civil War and all that , and here that
's not the case .
</item 10>
=== END ITEM id=162142f0ed611dab ===

=== ITEM id=d1e7434d40f8af7b ===
Gold sentences and facts:
Sentence 0: Year Film Role Notes Dumb and Dumber State Trooper Down Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr. Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P. Fudge My Life in Ruins Big Al Sawchuck 2009 The Haunted World of El Superbeasto Gerard the Exterminator ( voice ) 2011 Spooky Buddies Warwick the Warlock Lloyd the Conqueror Vulcan Dahmer Vs. Gacy God Dan Vs . Hiram Episode : Technology Robot Chicken Professor Cornelius Q. Quibblefingers ( voice ) 1 episode ; also writes 4 episodes 2012 The High Fructose Adventures of Annoying Orange Apple ( voice ) Robot and Monster Monster ( voice ) Main Role 2014 Back in the Day Skunk 2015 Jake and the Never Land Pirates Captain Frost ( voice ) Penn Zero : Part - Time Hero Helper Hue ( voice ) 2016 Sausage Party Ketchup Bottle / Drug Dealer / Baba Ganoush ( voice ) Skylanders Academy Hugo ( voice ) 2017 - present Puppy Dog Pals Bob ( voice ) Also show creator
  Fact 0.0: The listed performer had the role State Trooper in Dumb and Dumber.
  Fact 0.1: The listed performer appeared in Down Periscope, with no role given.
  Fact 0.2: The listed performer had the role 'Sonar' Lovacelli in E.T.
  Fact 0.3: The listed performer had the role Fred Z. Randall in RocketMan.
  Fact 0.4: The listed performer had the role Pet Wrangler in Wag the Dog.
  Fact 0.5: The listed performer had the role Himself in Space Ghost Coast to Coast, for 1 episode.
  Fact 0.6: In 1998, the listed performer had the role Kenny Davis in Half Baked.
  Fact 0.7: In 1998, the listed performer had the role Hitchhiker, uncredited, in There's Something About Mary.
  Fact 0.8: In 1998, the listed performer had the role Callum in Dog Park.
  Fact 0.9: In 1998, the listed performer had the role Newton, a voice role, in Ned's Newt.
  Fact 0.10: In 1998, the listed performer had the role Tucker in Mr. Headmistress.
  Fact 0.11: In 1999, the listed performer had the role Slater in Superstar.
  Fact 0.12: In 2000, the listed performer had the role Special Agent Steve Hanson in The Whole Nine Yards.
  Fact 0.13: In 2000, the listed performer had the role Officer Harry Cox in Big Money Hustlas.
  Fact 0.14: In 2000, the listed performer had the role Richard Breggs in Becoming Dick.
  Fact 0.15: In 2000, the listed performer had the role Darren in Freddy Got Fingered.
  Fact 0.16: In 2000, the listed performer had the role Mike Bonner, a voice role, in Gary & Mike.
  Fact 0.17: In 2002, the listed performer had the role Sheriff Hubbard in Back by Midnight.
  Fact 0.18: In 2002, the listed performer had the role Doofer / Roberta in Sorority Boys.
  Fact 0.19: In 2003, the listed performer had the role Jake in Family Tree.
  Fact 0.20: In 2003, the listed performer had the role Zee in Kart Racer.
  Fact 0.21: In 2005, the listed performer had the role Bleckman in Lucky 13.
  Fact 0.22: In 2005, the listed performer had the role Policeman in Because of Winn-Dixie.
  Fact 0.23: In 2005, the listed performer had the role Lugnut, a voice role, in Robots.
  Fact 0.24: In 2006, the listed performer had the role Russell Porpis-Gunders in Employee of the Month.
  Fact 0.25: In 2006, the listed performer had the role Rip in Surf School.
  Fact 0.26: In 2006, the listed performer had the role Head Dog Cop, a voice role, for 1 episode, in Ergo Proxy.
  Fact 0.27: In 2007, the listed performer had the role Carl, a voice role, in Meet the Robinsons.
  Fact 0.28: In 2007, the listed performer had the role Sheriff Rosco P. Coltrane in The Dukes of Hazzard: The Beginning.
  Fact 0.29: In 2008, the listed performer had the role Derek in Bachelor Party 2: The Last Temptation.
  Fact 0.30: In 2008, the listed performer had the role Giraffe, a voice role, in Madagascar: Escape 2 Africa.
  Fact 0.31: In 2009, the listed performer had the role Fudge Face Elmore P. Fudge in Fudgy Wudgy.
  Fact 0.32: In 2009, the listed performer had the role Big Al Sawchuck in My Life in Ruins.
  Fact 0.33: In 2009, the listed performer had the role Gerard the Exterminator, a voice role, in The Haunted World of El Superbeasto.
  Fact 0.34: In 2011, the listed performer had the role Warwick the Warlock, Lloyd the Conqueror, and Vulcan in Spooky Buddies.
  Fact 0.35: In 2011, the listed performer had the role God in Dahmer Vs. Gacy.
  Fact 0.36: In 2011, the listed performer had the role Robot in Dan Vs. Hiram, Episode: Technology.
  Fact 0.37: In 2011, the listed performer had the role Professor Cornelius Q. Quibblefingers, a voice role, for 1 episode, in Robot Chicken.
  Fact 0.38: The listed performer also wrote 4 episodes of Robot Chicken.
  Fact 0.39: In 2012, the listed performer had the role Apple, a voice role, in The High Fructose Adventures of Annoying Orange.
  Fact 0.40: In 2012, the listed performer had the role Monster, a voice role and a main role, in Robot and Monster.
  Fact 0.41: In 2014, the listed performer had the role Skunk in Back in the Day.
  Fact 0.42: In 2015, the listed performer had the role Captain Frost, a voice role, in Jake and the Never Land Pirates.
  Fact 0.43: In 2015, the listed performer had the role Helper Hue, a voice role, in Penn Zero: Part-Time Hero.
  Fact 0.44: In 2016, the listed performer had the role Ketchup Bottle / Drug Dealer / Baba Ganoush, a voice role, in Sausage Party.
  Fact 0.45: In 2016, the listed performer had the role Hugo, a voice role, in Skylanders Academy.
  Fact 0.46: From 2017 to the present, the listed performer had the role Bob, a voice role, in Puppy Dog Pals.
  Fact 0.47: The listed performer is also the creator of Puppy Dog Pals.

Retrieved items:
<item 1>
California Highway Patrol
They spotted the car , fell in behind , called for backup , and began the stop procedure . When the suspects ' vehicle came to a halt in the parking lot , the
driver was instructed to step out of the vehicle and spread his hands on the hood . Gore approached him and Frago moved to the passenger side . The passenger
side door suddenly swung open and the passenger sprung out , firing at Frago , who fell with two shots in his chest . The gunman , later identified as Jack
Twinning , then turned and fired once at Gore , who returned fire . In that moment the driver , Bobby Davis , turned and shot Gore twice at close range . Both
officers died instantly . When Officers James Pence and George Alleyn drove in moments later , they could not see suspects or other officers , but both
immediately came under fire . Pence put out an 11 ‐ 99 call ( `` officer down '' ) then took cover behind the passenger door . Alleyn grabbed his shotgun , and
positioned himself behind the driver - side door . Both officers were mortally wounded in the ensuing exchange , and one suspect was hit .
</item 1>

<item 2>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 2>

<item 3>
Harland Williams
Williams created the Disney Junior series Puppy Dog Pals that debuted in 2017 . Filmography ( edit ) Year Film Role Notes Dumb and Dumber State Trooper Down
Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny
Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr.
</item 3>

<item 4>
California Highway Patrol
`` I was upset there were n't four or five more rounds in there . After that , I ran and jumped in a ditch . The dumbest thing is , I still had the service
revolver in my hand . I was afraid when more police came they 'd think I was one of the gunman . So I put it behind me and said , ' They went that way . ' ''
Suspects Jack Twinning and Bobby Davis escaped , later abandoned their vehicle and then split up . For nine hours , officers blanketed the area searching for
the killers . Twinning broke into a house and briefly held a man hostage . Officers used tear gas before storming the house , but Twinning committed suicide
with the shotgun he stole from Frago . Davis was captured , stood trial and convicted on four counts of murder . He was sentenced to death , but in 1972 , the
California Supreme Court declared the death penalty to be cruel and unusual punishment and in 1973 , the court commuted his sentence to life in prison .
</item 4>

<item 5>
The Dukes of Hazzard (film)
Locations ( edit ) The majority of the film was shot in and around Clinton , Louisiana . The street scenes are set in Atlanta , but filmed in the New Orleans
Central Business District , and the university scenes were shot on the campus of Louisiana State University . Reception ( edit ) Box office ( edit ) The film
was number one at the box office its opening weekend and grossed $30.7 million on 3,785 screens . It also had an adjusted - dollar rank of number 24 all - time
for August releases . The film eventually collected $111 million worldwide , although it was much less successful financially outside the United States .
Critical reception ( edit ) Roger Ebert gave the film one star , calling it a `` lame - brained , outdated wheeze '' and wondered that Burt Reynolds ' part in
the film is `` karma-wise ... the second half of what Smokey and the Bandit was the first half of '' . Ebert also named it the second - worst film of the year
and Richard Roeper named it the worst film of 2005 .
</item 5>

<item 6>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 6>

<item 7>
Harland Williams
Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs
Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart
Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip
Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor
Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P.
</item 7>

<item 8>
California Highway Patrol
Peyer declined to provide a sample , nor explain why he declined to do so . Media references ( edit ) CHiPs was a fictional television drama show ( 1977 - 1983
) about the CHP , and the CHP helped out in the 1955 TV show Highway Patrol starring Broderick Crawford . Also the CHiPs TV series was made into a modern
television film called CHiPs ' 99 . A comical motion picture version of the show has been released in 2017 ( CHiPS ) but it has not been endorsed by the CHP at
all . Former CHP Commissioner Maury Hannigan was the host of the syndicated TV series Real Stories of the Highway Patrol , which aired in the mid-to - late
1990s . The show featured not just the CHP , but also the Minnesota State Patrol , the Nevada Highway Patrol , the Alaska State Troopers , the South Dakota
Highway Patrol , the Rhode Island State Police , the Massachusetts State Police , the West Virginia State Police and many other state police and highway patrol
agencies in the United State of America , and in one episode , the Royal Newfoundland Constabulary .
</item 8>

<item 9>
The Dukes of Hazzard (film)
The characters are now campus security officers , who give a warning to the Duke boys for driving too slowly . Paul Soter appears as TV newsman Rick Shakely ,
who is reporting from the Hazzard Road Rally . Charlie Finn , who played a dimwitted fast - food employee in Super Troopers , appears as a dimwitted geology
student who assists the Duke boys with the coal samples . Other notable cameos During the bar - fight scene , Indy - car driver A.J. Foyt IV appears as himself
. Bloopers roll under the ending credits , one of which features Rip Taylor interrupting the bedroom scene with Luke and the two college girls . Taylor had
previously appeared with Knoxville in Jackass : The Movie . Production ( edit ) Stunts ( edit ) Knoxville said he was initially reluctant to take on the role ,
but was persuaded by script changes and the presence of Dan Bradley as stunt coordinator and second unit for the car chase scenes . Knoxville praised him ,
saying , `` everyone in Hollywood wants Dan Bradley to shoot their car stuff '' . Locations ( edit ) The majority of the film was shot in and around Clinton ,
Louisiana .
</item 9>

<item 10>
The Dukes of Hazzard (film)
They are later arrested by Atlanta Police after running from campus police . Back in Hazzard , Daisy learns , with the help of Sheriff 's Deputy Enos Strate (
Michael Weston ) , that Billy Prickett has been hired by Boss Hogg to participate in the rally as a ringer . Boss Hogg then heads to Atlanta , where he informs
the Duke boys , in lock - up , that they are too late to stop him and reveals that the vote on Hogg 's proposition is at the same time as the rally , explaining
Billy Prickett 's involvement . During a transfer from detainment , Daisy helps the boys escape from the patrol car , and they speed home to try to inform the
townsfolk , escaping Atlanta Police , and the Georgia State Patrol after Bo outmaneuvers the city cops . Upon returning home , the Dukes discover that Boss Hogg
and Rosco had taken Uncle Jesse and Pauline hostage , an obvious trap for the boys , and that Billy is in on the scheme because he is ashamed of the town 's low
status .
</item 10>
=== END ITEM id=d1e7434d40f8af7b ===

=== ITEM id=a12984bb8b5f5639 ===
Gold sentences and facts:
Sentence 0: Urdu lyrics كشورِ حسين شاد باد ‬ پاک سرزمین شاد باد ‬ ارضِ پاکستان ! ‏ ‬ تُو نشانِ عزمِ عالی شان ‬ مرکزِ یقین شاد باد ‬ قوّتِ اُخوّتِ عوام ‬ پاک سرزمین کا نظام ‬ پائنده تابنده باد ! ‏ ‬ قوم ، ملک ، سلطنت ‬ شاد باد منزلِ مراد ‬ رہبرِ ترقّی و کمال ‬ پرچمِ ستاره و ہلال ‬ جانِ استقبال ! ‏ ‬ ترجمانِ ماضی ، شانِ حال ‬ سایۂ خدائے ذوالجلال ‬ Transliteration ( ALA - LC ) Translation ( poetical ) Translation ( literal ) Pāk sarzamīn shād bād Kishwar - i ḥasīn shād bād Tū nishān - i ʿazm - i ʿālī shān Arẓ - i Pākistān ! Markaz - i yaqīn shād bād May the holy land , stay glad ; Beautious realm , stay glad . Thou , the sign of high resolve -- O Land of Pakistan ! Citadel of faith , stay glad . Blessed be the sacred land , Happy be the bounteous realm . Thou symbol of high resolve , O Land of Pakistan ! Blessed be the citadel of faith . Pāk sarzamīn kā niz̤ām Quwwat - i Ukhuwwat - i ʿawām Qaum , mulk , salt̤anat Pāyindah tābindah bād ! Shād bād manzil - i murād Order of the holy land , Power of fraternity of the populace ; The nation , country , and domain ; Ever luminous remain ! The cherished goal , stay glad . The order of this sacred land , The might of the brotherhood of the people , May the nation , the country , and the state , Shine in glory everlasting ! Blessed be the goal of our ambition . Parcam - i sitārah o - hilāl Rahbar - i taraqqī o - kamāl Tarjumān - i māẓī , shān - i ḥāl Jān - i istiqbāl ! Sāyah - yi Khudā - yi Ẕū l - jalāl Flag with the star and crescent , The leader of progress and ascent , Dragoman of past , the pride of present ; Soul of the future ! Shadow of the God of grandeur . The flag of the crescent and star , Leads the way to progress and perfection , Interpreter of our past , glory of our present , inspiration for our future ! Shade of God , the Glorious and Mighty .
  Fact 0.0: The text gives Urdu lyrics about the Land of Pakistan.
  Fact 0.1: The text gives an ALA-LC transliteration of the Urdu lyrics.
  Fact 0.2: The text gives a poetical English translation of the lyrics.
  Fact 0.3: The text gives a literal English translation of the lyrics.
  Fact 0.4: The lyrics wish that the holy land stay glad.
  Fact 0.5: The lyrics wish that the beautiful realm stay glad.
  Fact 0.6: The lyrics describe the Land of Pakistan as the sign of high resolve.
  Fact 0.7: The lyrics describe the Land of Pakistan as the citadel of faith.
  Fact 0.8: The lyrics refer to the order of the holy land.
  Fact 0.9: The lyrics refer to the power of the brotherhood of the people.
  Fact 0.10: The lyrics wish that the nation, the country, and the state shine in everlasting glory.
  Fact 0.11: The lyrics wish that the cherished goal stay glad.
  Fact 0.12: The lyrics refer to the flag with the star and crescent.
  Fact 0.13: The lyrics describe the flag as the leader of progress and ascent.
  Fact 0.14: The lyrics describe the flag as an interpreter of the past.
  Fact 0.15: The lyrics describe the flag as the pride of the present.
  Fact 0.16: The lyrics describe the flag as the soul of the future.
  Fact 0.17: The lyrics describe the land as the shade of God, the Glorious and Mighty.

Retrieved items:
<item 1>
Qaumi Taranah
In 1950 , the impending state visit of the Shah of Iran added urgency to the matter and resulted in the Government of Pakistan asking the NAC to submit an
anthem without further delay . The NAC Chairman , then Federal Minister for Education , Fazlur Rahman , asked several poets and composers to write lyrics but
none of the submitted works were deemed suitable . The NAC also examined several different tunes and eventually selected the one presented by Ahmed G. Chagla
and submitted it for formal approval . On 21 August 1949 , the Government of Pakistan adopted Chagla 's tune for the national anthem . The anthem , without
lyrics , was performed for the first time for a foreign head of state on the state visit of the Shah of Iran to Pakistan in Karachi on 1 March 1950 by a
Pakistan Navy band . It was later played for Prime Minister Liaquat Ali Khan during his official visit to the United States on 3 May 1950 . It was played before
the NAC on 10 August 1950 . Official recognition to the national anthem , however , was not given until August 1954 .
</item 1>

<item 2>
Qaumi Taranah
rɑː. nɑ ) , lit . `` National Anthem '' ) , also known as Pāk Sarzamīn ( Urdu : پاک سرزمین ‬ ‎ , pronounced ( ˈpɑːk ˈsər. zə. miːn ) , lit . `` The Sacred Land
'' ) , is the national anthem of Pakistan . Its music was composed by Ahmad G. Chagla in 1949 , preceding the lyrics , which were written by Hafeez Jullundhri
in 1952 . It was officially adopted as Pakistan 's national anthem in August 1954 and was recorded in the same year by eleven major singers of Pakistan
including Ahmad Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar
Wasi Ali .
</item 2>

<item 3>
Qaumi Taranah
1954 -- Officially adopted as the national anthem and broadcast for the first time on Radio Pakistan on 13 August 1955 -- Sung by 11 famous Pakistani singers
including Ahmad Rushdi , Shamim Bano , Kokab Jehan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zwar Hussain , Akhtar Abbas , Ghulam Dastgir , Anwar Zaheer
and Akhtar Wassi 1996 -- Rendered in electric guitar for the first time by Pakistani rock band Junoon in their album Inquilaab 2009 -- Rendered as an acoustic
instrumental for the first time by Pakistani musician Jehangir Aziz Hayat 2011 -- On 14 August , 5,857 people gathered in a stadium in Karachi to sing the Qaumi
Taranah and set a new world record for most people gathered to sing a national anthem simultaneously . 2012 -- On 20 October , 42,813 people gathered in a
stadium in Lahore to sing the Qaumi Taranah and set a new world record for most people gathered to sing a national anthem simultaneously , which was certified
by Guinness World Records .
</item 3>

<item 4>
Qaumi Taranah
Contents ( hide ) 1 History 2 Music 3 Lyrics 4 Milestones 5 See also 6 References 7 External links History ( edit ) In early 1948 , A.R. Ghani , a Muslim from
Transvaal , South Africa , offered two prizes of five thousand rupees each for the poet and composer of a new national anthem for the newly independent state of
Pakistan . The prizes were announced through a government press advertisement published in June 1948 . In December 1948 , the Government of Pakistan established
the National Anthem Committee ( NAC ) with the task of coming up with the composition and lyrics for the official national anthem of Pakistan . The NAC was
initially chaired by the Information Secretary , Sheikh Muhammad Ikram , and its members included several politicians , poets and musicians , including Abdur
Rab Nishtar , Ahmad G. Chagla and Hafeez Jullundhri . The NAC encountered early difficulties in finding suitable music and lyrics . When President Sukarno of
Indonesia became the first foreign head of state to visit Pakistan on 30 January 1950 , there was no Pakistani national anthem to be played .
</item 4>

<item 5>
Qaumi Taranah
Qaumi Taranah - wikipedia Qaumi Taranah Jump to : navigation , search Qaumi Taranah قومی ترانہ English : National Anthem of Pakistan National Anthem National
anthem of Pakistan Lyrics Hafeez Jullundhri , June 1952 Music Ahmad G. Chagla , 21 August 1949 Adopted 13 August 1954 Relinquished Present Audio sample Qaumi
Taranah ( Instrumental ) file help Ahmad G. Chagla composed the music of the National Anthem of Pakistan in 1949 Ahmed Rushdi recorded the National Anthem of
Pakistan in 1954 The Qaumi Taranah ( Urdu : قومی ترانہ ‬ ‎ , Qaumī Tarānah pronounced ( ˈqɔː. mi ˈt̪ə. rɑː. nɑ ) , lit .
</item 5>

<item 6>
Qaumi Taranah
Archived from the original on 2007 - 10 - 26 . Retrieved 2007 - 12 - 09 . National Anthem of Pakistan Pakistan National Songs National symbols of Pakistan Main
symbols Qaumi Taranah ( national anthem ) State emblem of Pakistan Flag of Pakistan Iman , Ittihad , Nazm ( motto ) Lahore Resolution ( national document ) Urdu
( national language ) People Muhammad Ali Jinnah ( Father of the Nation ) Muhammad Iqbal ( National poet ) Fatima Jinnah ( Mother of the Nation ) Other symbols
Cedrus deodara ( national tree ) Chukar partridge ( national bird ) Jasminum officinale ( national flower ) Markhor ( national animal ) Structures Bab -
e-Khyber ( monument ) Faisal Mosque Mazar - e-Quaid ( mausoleum ) Minar - e-Pakistan ( monument ) Pakistan Monument ( monument ) Quaid - e-Azam Residency
National anthems of Asia National Abkhazia Afghanistan Armenia Artsakh ( Nagorno - Karabakh ) Azerbaijan Bahrain Bangladesh Bhutan Brunei Cambodia China Cyprus
East Timor Egypt Georgia India Indonesia Iran Iraq Israel Japan Jordan Kazakhstan North Korea South Korea
</item 6>

<item 7>
Qaumi Taranah
BBC News . 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ `` Indian A capella group sings Pakistan 's national anthem on 71st Independence Day '' .
Firstpost. 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ Staff , Images ( 2017 - 08 - 12 ) . `` Indians sing the Pakistani national anthem as a gift for
our Independence Day '' . Images . Retrieved 2017 - 08 - 15 . External links ( edit ) This article contains Urdu text . Without proper rendering support , you
may see unjoined letters running left to right or other symbols instead of Urdu script . Wikimedia Commons has media related to : Qaumi Taranah ( category ) Pak
Sar Zameen Shad Bad Lyrics Pakistan : Qaumī Tarānah -- Audio of the national anthem of Pakistan , with information and lyrics Ministry of Information and
Broadcasting , Government of Pakistan . `` National Anthem '' . Archived from the original on 2007 - 10 - 26 .
</item 7>

<item 8>
Qaumi Taranah
Official recognition to the national anthem , however , was not given until August 1954 . The NAC distributed records of the composed tune amongst prominent
poets , who responded by writing and submitting several hundred songs for evaluation by the NAC . Eventually , the lyrics written by Hafeez Jullundhri were
approved and the new national anthem was broadcast publicly for the first time on Radio Pakistan on 13 August 1954 , sung by Hafeez Jullundhri himself .
Official approval was announced by the Ministry of Information and Broadcasting on 16 August 1954 . The composer , Ahmed G. Chagla , died in 1953 , before the
new national anthem was officially adopted . In 1955 , there was a performance of the national anthem involving 11 major singers of Pakistan , including Ahmad
Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar Wasi Ali .
</item 8>

<item 9>
Qaumi Taranah
Retrieved 1 September 2016 . Jump up ^ National Anthem. http://anisshakur.tripod.com/id129.html Jump up ^ Michael Jamieson Bristow , National-Anthems.org . ``
Forty National Anthems '' . Retrieved 2006 - 04 - 12 . ^ Jump up to : Aqeel Abbas Jafari ( 2010 ) . Pakistan Chronicle ( in Urdu ) ( 1st ed . ) . 94 / 1 , 26th
St. , Ph. 6 , D.H.A. , Karachi , Pakistan : Wirsa Publishers . p. 42 . ISBN 9789699454004 . ^ Jump up to : `` National Anthem '' . Ministry of Information and
Broadcasting , Government of Pakistan . Archived from the original on 2007 - 10 - 26 . Retrieved 2007 - 11 - 29 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` Ahmad
Rushdi '' . http://mazhar.dk/film/singers/ahmadrushdi/ .
</item 9>

<item 10>
Qaumi Taranah
http://mazhar.dk/film/singers/ahmadrushdi/ . Retrieved 2006 - 04 - 12 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` National Anthem of Pakistan '' . Retrieved 2006
- 04 - 12 . Jump up ^ Information Ministry , Government of Pakistan . `` Basic Facts '' . Retrieved 2006 - 04 - 12 . Jump up ^ ( 1 ) , National Anthem of
Pakistan Jump up ^ ( 2 ) , The National Anthem of Pakistan Jump up ^ ( 3 ) , Switch to Urdu from English as official language a hurdle for Pakistan Jump up ^ (
4 ) , History can not be falsely written and manufactured Jump up ^ ( 5 ) , Pakistan Basic Facts Jump up ^ Pasha , Muhammad A. English Composition ( Part II ) .
Lahore : Command Publications . Jump up ^ `` Pakistan creates new anthem record '' . The Express Tribune . 2011 - 08 - 12 . Retrieved 2012 - 01 - 01 . Jump up ^
`` Pakistan reclaims anthem singing record '' .
</item 10>
=== END ITEM id=a12984bb8b5f5639 ===

=== ITEM id=c7d6fe5bdf7b91e5 ===
Gold sentences and facts:
Sentence 0: Duncan 's diagnosis was publicly confirmed during a CDC news conference the same day . That evening , Duncan reported feeling better and requested to watch a movie . The following morning , Duncan was breathing rapidly and complaining of `` pain all over '' . By the afternoon , however , he was able to eat , and the doctor noted that he was feeling better . The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola virus disease and the index patient for the later infections of nurses Pham and Vinson .
  Fact 0.0: The CDC confirmed Duncan's Ebola diagnosis at a news conference on the same day.
  Fact 0.1: That evening, Duncan reported feeling better and requested to watch a movie.
  Fact 0.2: The following morning, Duncan was breathing rapidly and complaining of 'pain all over'.
  Fact 0.3: By the afternoon, Duncan was able to eat.
  Fact 0.4: By the afternoon, the doctor noted that Duncan was feeling better.
  Fact 0.5: On October 3, Duncan again reported feeling abdominal pain.
  Fact 0.6: That evening, October 3, the hospital contacted Chimerix.
  Fact 0.7: Chimerix is a biotechnology company developing Brincidofovir to combat the disease.
  Fact 0.8: On October 4, Duncan's organs were failing.
  Fact 0.9: On October 4, Duncan was intubated to help him breathe.
  Fact 0.10: On October 4 in the afternoon, the hospital began administering Brincidofovir to Duncan.
  Fact 0.11: Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock.
  Fact 0.12: On October 7, the hospital reported that Duncan's condition was improving.
  Fact 0.13: Duncan died at 7:51 a.m. on October 8.
  Fact 0.14: Duncan became the first person to die in the United States of Ebola virus disease.
  Fact 0.15: Duncan was the index patient for the later infections of nurses Pham and Vinson.
Sentence 1: In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola '' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first was reported in September 2014 . Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered .
  Fact 1.0: In December 2014, Ebola virus cases occurred in the United States due to four laboratory-confirmed cases of Ebola virus disease in the United States.
  Fact 1.1: Ebola virus disease is commonly known as Ebola.
  Fact 1.2: Eleven Ebola cases have been reported.
  Fact 1.3: The eleven reported cases include the four cases diagnosed in the United States and seven cases medically evacuated from other countries.
  Fact 1.4: The first of the eleven cases was reported in September 2014.
  Fact 1.5: Nine of the eleven cases contracted the disease outside the US and traveled into the country.
  Fact 1.6: The nine people traveled into the country either as regular airline passengers or as medical evacuees.
  Fact 1.7: Two of those nine people died.
  Fact 1.8: Two people have contracted Ebola in the United States.
  Fact 1.9: Both people who contracted Ebola in the United States were nurses who treated an Ebola patient.
  Fact 1.10: Both nurses have recovered.
Sentence 2: On September 30 , 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 , Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were :
  Fact 2.0: The CDC announced on September 30, 2014 that Thomas Eric Duncan had been diagnosed with Ebola in Dallas, Texas.
  Fact 2.1: Thomas Eric Duncan was reportedly 42 years old.
  Fact 2.2: CDC reports later corrected Thomas Eric Duncan's age to 45 years old.
  Fact 2.3: Thomas Eric Duncan was a Liberian national.
  Fact 2.4: Thomas Eric Duncan was visiting the United States from Liberia.
  Fact 2.5: Thomas Eric Duncan had been visiting family in Dallas.
  Fact 2.6: Thomas Eric Duncan was treated at Texas Health Presbyterian Hospital Dallas.
  Fact 2.7: By October 4, Thomas Eric Duncan's condition had deteriorated from 'serious but stable' to 'critical'.
  Fact 2.8: Thomas Eric Duncan died of Ebola on October 8.
  Fact 2.9: Three other cases were diagnosed in the United States as of October 2014, and they are listed after this sentence.
Sentence 3: Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
  Fact 3.0: The page is about Ebola virus disease in the U.S.
  Fact 3.1: The page includes a map of Ebola cases and infrastructure throughout the U.S.
  Fact 3.2: The page has categories for cases contracted in the U.S., cases first diagnosed in the U.S., and cases evacuated to the U.S. from other countries.
  Fact 3.3: The table lists 7 cases evacuated to the U.S. from other countries.
  Fact 3.4: The table lists 11 total cases.
  Fact 3.5: The table has a 'Deaths' label.
  Fact 3.6: The table lists 9 recoveries from Ebola.
  Fact 3.7: The table lists 0 active cases.

Retrieved items:
<item 1>
Ebola virus cases in the United States
Ebola virus cases in the United States - wikipedia Ebola virus cases in the United States Jump to : navigation , search This article is about Ebola virus cases
in the United States starting in 2014 . For the 2013 - 2016 outbreak in Africa , see Ebola virus epidemic in West Africa . For the U.S. discovery of a different
species of the Ebolavirus genus in 1989 , see Reston virus . Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases
contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola
'' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first
was reported in September 2014 .
</item 1>

<item 2>
Ebola virus cases in the United States
( edit ) Articles related to the West African Ebola virus epidemic Overview Ebola virus disease Timeline of the epidemic Responses to the epidemic Timeline ,
tables and graphs Ebola virus disease treatment research Post-Ebola virus syndrome Ebola vaccine Nations with widespread cases Guinea Liberia Sierra Leone Other
affected nations Mali Nigeria Senegal Spain United States United Kingdom Italy Other outbreaks List of Ebola outbreaks 1976 Zaire outbreak 2014 DR Congo
outbreak List of epidemics First case : Thomas Eric Duncan ( edit ) Main article : Thomas Eric Duncan Thomas Eric Duncan in Liberia ( edit ) Thomas Eric Duncan
was from Monrovia , Liberia , to date the country hit hardest by the Ebola virus epidemic . Duncan worked as a personal driver for the general manager of
Safeway Cargo , a FedEx contractor in Liberia . According to manager Henry Brunson , Duncan had abruptly quit his job on September 4 , 2014 , giving no reason .
On September 15 , 2014 , the family of Marthalene Williams , who later died of Ebola virus disease , could not call an ambulance to transfer the pregnant
Williams to a hospital . Duncan , their tenant , helped to transfer Williams by taxi to an Ebola treatment ward in Monrovia .
</item 2>

<item 3>
Ebola virus cases in the United States
Medical evacuations from West Africa ( edit ) Local transmissions -- no deaths : Texas Initial cases -- deaths : Texas Initial cases -- no deaths : New York
Medically evacuated cases -- no deaths : Georgia , Maryland Medically evacuated cases -- deaths : Nebraska By the time the Ebola virus epidemic in West Africa
was over in June 2016 , seven Americans had been evacuated to the U.S. for treatment after contracting Ebola virus while working with medical teams trying to
stop the epidemic . Kent Brantly , a physician and medical director in Liberia for the aid group Samaritan 's Purse , and co-worker Nancy Writebol were infected
in July 2014 , while working in Monrovia . Both were flown to the United States at the beginning of August via the CDC 's Aeromedical Biological Containment
System for further treatment in Atlanta 's Emory University Hospital . On August 21 , Brantly and Writebol recovered and were discharged . Brantly has since
donated blood to three others with Ebola ( Sacra , Mukpo and Pham ) .
</item 3>

<item 4>
Ebola virus cases in the United States
WYFF4 . Retrieved October 3 , 2014 . Jump up ^ `` First US Ebola case Thomas Duncan ' critical ' '' . BBC News . October 4 , 2014 . Retrieved October 4 , 2014 .
Jump up ^ `` Texas Ebola patient dies '' . CNN . October 8 , 2014 . Retrieved October 8 , 2014 . ^ Jump up to : Fernandez , Manny ( October 12 , 2014 ) . ``
Texas Health Worker Tests Positive for Ebola '' . New York Times . Retrieved October 12 , 2014 . Jump up ^ `` Ebola outbreak : Second Texas health worker '
tests positive ' '' . BBC News . October 15 , 2014 . Retrieved October 15 , 2014 . ^ Jump up to : `` New York Doctor Just Back From Africa Has Ebola '' . NBC
News . October 23 , 2014 . Retrieved October 23 , 2014 . Jump up ^ Dan Diamond ( October 20 , 2014 ) .
</item 4>

<item 5>
Ebola virus cases in the United States
In addition , on September 21 , a CDC employee , who showed no symptoms of the disease , was flown back to the United States as a preventative measure after low
- risk exposure with a healthcare worker ; he posed no risks to his family or the United States , and he is not known to have died . On September 28 , a fourth
American doctor was admitted to National Institutes of Health hospital . On October 2 , NBC News photojournalist Ashoka Mukpo , covering the outbreak in Liberia
, tested positive for Ebola after showing symptoms . Four other members of the NBC team , including physician Nancy Snyderman , were being closely monitored for
symptoms . Mukpo was evacuated on October 6 to the University of Nebraska Medical Center for treatment in their isolation unit . On October 21 , Mukpo was
declared Ebola - free and allowed to return to his home in Rhode Island . A government official in Sierra Leone announced on November 13 , that a doctor from
Sierra Leone , a permanent resident of the United States married to a U.S. citizen , would be transported to the Nebraska Medical Center for treatment for Ebola
. The doctor , identified later as Dr.
</item 5>

<item 6>
Ebola virus cases in the United States
See also ( edit ) Ebola virus epidemic in Guinea Ebola virus epidemic in Liberia Ebola virus epidemic in Nigeria Ebola virus epidemic in Sierra Leone Ebola
virus disease in Spain List of Ebola outbreaks West Nile virus in the United States Notes ( edit ) Jump up ^ Duncan contracted Ebola in Liberia and was
diagnosed in the U.S. Two of the victims had had direct contact with Duncan . One victim was diagnosed in New York City ; his case is unrelated to the Texas
cases . ^ Jump up to : This takes into account the seven Americans who were medically evacuated from West Africa with Ebola . References ( edit ) Jump up ^ ``
Who are the American Ebola patients ? '' , CNN , October 6 , 2014 Jump up ^ `` Nebraska Hospital : Surgeon With Ebola Has Died '' . ABC News . November 17 ,
2014 . Retrieved November 17 , 2014 . Jump up ^ `` 2014 Ebola Outbreak - Case Count '' . CDC . December 15 , 2014 . Retrieved December 16 , 2014 . Jump up ^ ``
First diagnosed case of Ebola in the U.S. '' CNN .
</item 6>

<item 7>
Ebola virus cases in the United States
Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those
nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered . On September 30
, 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national
visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at
Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 ,
Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were : October 11 , 2014 , a nurse , Nina Pham , who had provided
care to Duncan at the hospital . October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan .
</item 7>

<item 8>
Ebola virus cases in the United States
October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan . October 23 , 2014 , physician Craig Spencer , diagnosed in New York City ; he had just
returned from working with Doctors Without Borders in Guinea , a country in west Africa . Hundreds of people were tested or monitored for potential Ebola virus
infection , but the two nurses were the only confirmed cases of locally transmitted Ebola . Public health experts and the Obama administration opposed
instituting a travel ban on Ebola endemic areas , stating that it would be ineffective and would paradoxically worsen the situation . No one both contracted and
died of Ebola virus disease while in the United States . No new cases have been diagnosed in the United States since Craig Spencer was released from a New York
City hospital on November 11 , 2014 . Contents ( hide ) 1 Cases diagnosed in U.S. 1.1 First case : Thomas Eric Duncan 1.1. 1 Thomas Eric Duncan in Liberia 1.1.
2 Duncan 's illness in Dallas 1.1. 3 Contact tracing 1.1.
</item 8>

<item 9>
Ebola virus cases in the United States
These include the University of Nebraska Medical Center in Omaha , Nebraska , the National Institutes of Health in Bethesda , Maryland , St. Patrick Hospital
and Health Sciences Center in Missoula , Montana and Emory University Hospital in Atlanta , Georgia . Experimental treatments ( edit ) Main article : Ebola
virus disease treatment research There is as yet no FDA - approved medication or vaccine . The director of the US National Institute of Allergy and Infectious
Diseases , Dr. Anthony Fauci , has stated that the scientific community is still in the early stages of understanding how infection with the Ebola virus can be
treated and prevented . There is no cure or specific treatment that is currently approved . However , survival is improved by early supportive care with
rehydration and symptomatic treatment . Treatment is primarily supportive in nature . A number of experimental treatments are being considered for use in the
context of this outbreak , and are currently or will soon undergo clinical trials . In late August 2014 , both Kent Brantly and Nancy Writebol became the first
people to be given the experimental drug ZMapp . They both recovered , but there was no confirmation or proof that the drug was a factor .
</item 9>

<item 10>
Ebola virus cases in the United States
gov Family Filoviridae Cuevavirus Species Lloviu cuevavirus ( LLOV ) Ebolavirus Species Bundibugyo ebolavirus ( BDBV ) Reston ebolavirus ( RESTV ) Sudan
ebolavirus ( SUDV ) Taï Forest ebolavirus ( TAFV ) Zaire ebolavirus ( EBOV ) Outbreaks 1976 Sudan outbreak 1976 Zaire outbreak 2013 − 2016 West African Ebola
virus epidemic Timeline Reported cases and deaths Responses United Nations Ebola Response Fund Operation United Assistance in Guinea in Liberia in Mali in
Nigeria in Sierra Leone in Spain in the US in the UK Ouse to Ouse Tock Womey massacre 2014 DR Congo outbreak 2017 DR Congo outbreak Drug candidates BCX4430
Brincidofovir DZNep Favipiravir FGI - 103 FGI - 104 FGI - 106 JK - 05 Lamivudine TKM - Ebola ( failed ) Triazavirin ZMapp Vaccines cAd3 - ZEBOV rVSV - ZEBOV
Notable people William Close Peter Piot Selected patients
</item 10>
=== END ITEM id=c7d6fe5bdf7b91e5 ===

=== ITEM id=7fcbc15aba0aae52 ===
Gold sentences and facts:
Sentence 0: `` Lord Have Mercy on the Working Man '' is a song written by Kostas and recorded by American country music singer Travis Tritt . It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 .
  Fact 0.0: Lord Have Mercy on the Working Man is a song.
  Fact 0.1: Kostas wrote the song Lord Have Mercy on the Working Man.
  Fact 0.2: Travis Tritt, an American country music singer, recorded the song Lord Have Mercy on the Working Man.
  Fact 0.3: Lord Have Mercy on the Working Man was released in August 1992.
  Fact 0.4: Lord Have Mercy on the Working Man was the first of five singles from Travis Tritt's third studio album T-R-O-U-B-L-E.
  Fact 0.5: Lord Have Mercy on the Working Man became Travis Tritt's tenth entry on the Billboard Hot Country Singles & Tracks charts.
  Fact 0.6: The Billboard Hot Country Singles & Tracks chart is now called Hot Country Songs.
  Fact 0.7: Lord Have Mercy on the Working Man peaked at number 5 on the Billboard chart.
Sentence 1: `` Lord Have Mercy on the Working Man '' Single by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology `` Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' ( 1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 )
  Fact 1.0: Lord Have Mercy on the Working Man is a single by Travis Tritt from the album T-R-O-U-B-L-E.
  Fact 1.1: The B-side of the single is Lord Have Mercy on the Working Man (album version).
  Fact 1.2: The single was released on August 10, 1992.
  Fact 1.3: The single format is CD single.
  Fact 1.4: The genre of the single is country.
  Fact 1.5: The album version of the song has a length of 4:55.
  Fact 1.6: The label of the single is Warner Bros. Nashville.
  Fact 1.7: Kostas is the songwriter of Lord Have Mercy on the Working Man.
  Fact 1.8: Gregg Brown is the producer of Lord Have Mercy on the Working Man.
  Fact 1.9: The Travis Tritt singles chronology lists Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).
  Fact 1.10: The Travis Tritt singles chronology then repeats Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).

Retrieved items:
<item 1>
Lord Have Mercy on the Working Man
Critical reception ( edit ) Teresa M. Walker , in her review for the Gainesville Sun , said that with the assistance from superstars on the final chorus , the
song `` should shoot up the charts . '' Dave Larsen of the Dayton Daily News cited it as one of the stronger tracks on the album , saying that the album ``
works best when Tritt sticks with the populist approach . '' Alanna Nash of Entertainment Weekly said of the song , `` Tritt finds a solid image for his laborer
's lament ( ... ) But the melody is so slight that he resorts to an acoustic arrangement that evokes Jimmie Rodgers and the young Roy Acuff . '' Deborah Evans
Price , of Billboard magazine gave the song a mixed review , saying that while the final chorus of song features famous country artists , it was too bad that
the entire song is n't as strong as the last verse . Music video ( edit ) Jack Cole directed the song 's music video .
</item 1>

<item 2>
Graph query row:
song: Lord Have Mercy on the Working Man
relationship: CREATED
songwriter: Kostas
</item 2>

<item 3>
Lord Have Mercy on the Working Man
It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard
Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 . Contents ( hide ) 1 Content 2 Critical reception 3 Music video 4
Chart performance 5 References 6 External links Content ( edit ) `` Lord Have Mercy on the Working Man '' is a moderate up - tempo whose lyrics centralize on a
theme of economic injustice towards blue collar workers . The instrumentation features various forms of percussion from Sam Bacco , including crotales , wobble
board , spoons and a broom . Richard Bennett and Wendell Cox play guitar solos before the third verse , and Brooks & Dunn , T. Graham Brown , George Jones ,
Little Texas , Dana McVicker , Tanya Tucker and Porter Wagoner all sing background vocals on the final chorus .
</item 3>

<item 4>
Lord Have Mercy on the Working Man
Chicago Tribune . Retrieved 28 August 2009 . ^ Jump up to : Whitburn , Joel ( 2006 ) . The Billboard Book of Top 40 Country Hits ( 2 ed . ) . Billboard Books .
p. 353 . ISBN 0 - 8230 - 8291 - 1 . Jump up ^ Walker , Teresa M. ( 4 September 1992 ) . `` Travis Tritt 's rockin ' country '' . Gainesville Sun . Archived from
the original on 12 July 2012 . Retrieved 28 August 2009 . Jump up ^ Larsen , Dave ( 18 September 1992 ) . `` Recordings on review '' . Dayton Daily News .
Retrieved 28 August 2009 . Jump up ^ Nash , Alanna ( 4 September 1992 ) . `` T-R-O-U-B-L-E in mind '' . Entertainment Weekly . Retrieved 28 August 2009 . Jump
up ^ Billboard , August 15 , 1992 Jump up ^ Ellison , Curtis W.
</item 4>

<item 5>
Lord Have Mercy on the Working Man
On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lord_Have_Mercy_on_the_Working_Man&oldid=789364103 '' Categories : 1992 singles
Travis Tritt songs Songs written by Kostas ( songwriter ) Warner Bros. Records singles 1992 songs Hidden categories : Articles with hAudio microformats
Singlechart usages for Canadacountry Singlechart called without artist Singlechart called without song Singlechart usages for Billboardcountrysongs Talk
Contents About Wikipedia Add links This page was last edited on 6 July 2017 , at 22 : 41 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
Lord Have Mercy on the Working Man
Ai n't Country '' My Honky Tonk History `` The Girl 's Gone Wild '' `` What Say You '' ( with John Mellencamp ) `` I See Me '' The Storm `` You Never Take Me
Dancing '' `` Something Stronger Than Me '' The Calm After ... `` Sometimes Love Just Ai n't Enough '' ( With Tyler Reese ) Other songs `` This One 's Gonna
Hurt You ( For a Long , Long Time ) '' ( with Marty Stuart ) `` Take It Easy '' `` The Devil Comes Back to Georgia '' ( With Mark O'Connor , Charlie Daniels ,
Johnny Cash , and Marty Stuart ) `` Honky Tonkin 's What I Do Best '' ( With Marty Stuart ) `` Hope '' `` Here 's Your Sign ( Get the Picture ) '' ( With Bill
Engvall ) `` Same Old Train '' `` Out of Control Raging Fire '' ( with Patty Loveless ) `` Southern Boy '' ( with Charlie Daniels ) `` Bible Belt '' ( with
Little Feat ) `` Move It On Over '' `` Lonesome , On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Lord Have Mercy on the Working Man
Music video ( edit ) Jack Cole directed the song 's music video . In the book Country Music Culture : From Hard Times to Heaven , Curtis W. Ellison wrote that
the song 's music video `` confronted a litany of personal oppressions attributed to government policy '' that coincided with Bill Clinton 's presidential
campaign . Chart performance ( edit ) `` Lord Have Mercy on the Working Man '' spent 20 weeks on the Billboard Hot Country Singles & Tracks ( now Hot Country
Songs ) charts in 1992 , peaking at number 5 . The song also reached number 10 on the Canadian RPM Country Tracks charts . Chart ( 1992 ) Peak position Canada
Country Tracks ( RPM ) 10 US Hot Country Songs ( Billboard ) 5 References ( edit ) ^ Jump up to : T-R-O-U-B-L-E ( cassette insert ) . Travis Tritt . Warner
Bros. Records . 1992 . 45048 . Jump up ^ Hurst , Jack ( 13 August 1992 ) . `` Satellite show Travis Tritt to launch new album from Park West '' . Chicago
Tribune . Retrieved 28 August 2009 .
</item 7>

<item 8>
Lord Have Mercy on the Working Man
( 1995 ) . Country Music Culture : From Hard Times to Heaven . Univ . Press of Mississippi . p. 233 . ISBN 0 - 87805 - 722 - 6 . Retrieved 28 August 2009 . Jump
up ^ `` Top RPM Country Tracks : Issue 1864 . '' RPM . Library and Archives Canada . November 28 , 1992 . Retrieved August 15 , 2013 . Jump up ^ `` Travis Tritt
Chart History ( Hot Country Songs ) '' . Billboard . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Travis Tritt singles Country Club ``
Country Club '' `` Help Me Hold On '' `` I 'm Gonna Be Somebody '' `` Put Some Drive in Your Country '' `` Drift Off to Dream '' It 's All About to Change ``
Here 's a Quarter ( Call Someone Who Cares ) '' `` Anymore '' `` The Whiskey Ai n't Workin ' '' ( With Marty Stuart ) `` Nothing Short of Dying '' T-R-O-U-B-L-E
`` Lord Have Mercy on the Working Man '' ``
</item 8>

<item 9>
Lord Have Mercy on the Working Man
Lord Have Mercy on the Working Man - wikipedia Lord Have Mercy on the Working Man Jump to : navigation , search `` Lord Have Mercy on the Working Man '' Single
by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single
Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology ``
Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' (
1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Lord Have Mercy on the Working Man '' is a song
written by Kostas and recorded by American country music singer Travis Tritt .
</item 9>

<item 10>
Lord Have Mercy on the Working Man
Dying '' T-R-O-U-B-L-E `` Lord Have Mercy on the Working Man '' `` Can I Trust You with My Heart '' `` T-R-O-U-B-L-E '' `` Looking Out for Number One '' ``
Worth Every Mile '' Ten Feet Tall and Bulletproof `` Foolish Pride '' `` Ten Feet Tall and Bulletproof '' `` Between an Old Memory and Me '' `` Tell Me I Was
Dreaming '' Greatest Hits : From the Beginning `` Sometimes She Forgets '' `` Only You ( And You Alone ) '' The Restless Kind `` More Than You 'll Ever Know ''
`` Where Corn Do n't Grow '' `` She 's Going Home with Me '' `` Helping Me Get Over You '' ( With Lari White ) `` Still in Love with You '' No More Looking over
My Shoulder `` If I Lost You '' `` No More Looking Over My Shoulder '' `` Start the Car '' Down the Road I Go `` Best of Intentions '' `` It 's a Great Day to
Be Alive '' `` Love of a Woman '' `` Modern Day Bonnie and Clyde '' Strong Enough `` Strong Enough to Be Your Man '' `` Country Ai n't Country '' My Honky Tonk
History `` The Girl 's Gone Wild '' ``
</item 10>
=== END ITEM id=7fcbc15aba0aae52 ===
