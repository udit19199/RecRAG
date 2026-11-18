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

=== ITEM id=d933e8ab223bf087 ===
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
097 2008 December 30 7 : 00 PM Rice 38 Western Michigan 14 58,880 2009 December 31 2 : 30 PM Navy 35 Missouri 13 69,441 ESPN December 29 5 : 00 PM Illinois 38
Baylor 14 68,211 2011 December 31 11 : 00 AM Texas A&M 33 Northwestern 22 68,395 2012 December 28 8 : 00 PM Texas Tech 34 Minnesota 31 50,386 2013 December 27 5
: 00 PM Syracuse 21 Minnesota 17 32,327 2014 December 29 8 : 00 PM Arkansas 31 Texas 7 71,115 2015 December 29 8 : 00 PM LSU 56 Texas Tech 27 71,307 2016
December 28 8 : 00 PM Kansas State 33 Texas A&M 28 68,412 2017 December 27 8 : 00 PM Texas 33 Missouri 16 67,
</item 1>

<item 2>
Texas Bowl
Since 2017 , the game has been sponsored by Academy Sports + Outdoors . Contents ( hide ) 1 History 1.1 Replacing the Houston Bowl 1.2 Texas Bowl introduction
1.3 Sponsors 1.4 Conference tie - ins 1.5 Issues 2 Game results 3 MVPs 4 Most appearances 5 Appearances by conference 6 See also 7 References 8 External links
History ( edit ) Replacing the Houston Bowl ( edit ) Speculation surfaced questioning the long - term survival of the former Houston Bowl . The three - year
contract with EV1.net expired on December 31 , 2005 , leaving the bowl game without a title sponsor . A college football official told the Houston Chronicle
that the bowl was in danger of ceasing operations , as a result of the game losing its title sponsor and because the Houston Bowl still owed roughly $600,000 to
the Big 12 and Mountain West conferences following the 2005 game . However , the NCAA approved Lone Star Sports & Entertainment , a division of the Houston
Texans , who also play in Reliant Stadium , to take over game management .
</item 2>

<item 3>
Texas Bowl
Texas Bowl - wikipedia Texas Bowl Jump to : navigation , search `` AdvoCare V100 Bowl '' and `` AdvoCare Bowl '' redirect here . For the bowl game that AdvoCare
previously sponsored ( 2009 -- 13 ) , see Independence Bowl . Texas Bowl Academy Sports + Outdoors Texas Bowl Stadium NRG Stadium Location Houston , Texas
Operated 2006 -- present Conference tie - ins Big 12 , SEC Previous conference tie - ins Big East / C - USA / MWC ( alternating years ) ( 2006 -- 2009 ) Big Ten
( 2010 -- 2013 ) Payout US $ 3 million per team ( as of 2015 ) Preceded by Houston Bowl Sponsors Meineke Car Care ( 2011 -- 2012 ) AdvoCare ( 2014 -- 2016 )
Academy Sports + Outdoors ( 2017 - present ) Former names Texas Bowl ( 2006 -- 2010 ) Meineke Car Care Bowl of Texas ( 2011 -- 2012 ) Texas Bowl ( 2013 )
AdvoCare V100 Texas Bowl ( 2014 -- 2016 ) 2016 matchup Texas A&M vs.
</item 3>

<item 4>
Texas Bowl
In July 2006 , the NFL Network acquired TV rights and naming rights to the bowl . Texas Bowl introduction ( edit ) The Texas Bowl name and logo were officially
unveiled on August 10 , 2006 , at a press conference along with conference affiliations for the bowl spots . The Big 12 , Big East and Conference USA will be
affiliated with the game , as well as Texas Christian University of the Mountain West . The 2006 matchup featured teams from the Big 12 and Big East Conferences
. On December 3 , 2006 , Rutgers accepted an invitation to play Kansas State in the inaugural Texas Bowl . `` We 're ecstatic about having Rutgers , '' Texas
Bowl director David Brady said . `` This is a top - 15 team that was three yards away from a BCS game . We could n't be happier to have them here . '' 2010
marked the eleventh consecutive year a bowl game has played in Houston , and the 40th year overall with a bowl game there ( the Bluebonnet Bowl lasted 29 years
) .
</item 4>

<item 5>
Texas Bowl
In 2007 , TCU took the place of the Big 12 team when Kansas and Oklahoma were put into the BCS , and Houston , a `` home team , '' represented C - USA . The
conferences would receive $612,500 each as per the rules of the agreements as usually , the Big East ( or Big 12 ) would have received $750,000 for playing and
C - USA would have received a $500,000 stipend for their team playing . Issues ( edit ) According to Sports Illustrated , in 2008 the bowl required Western
Michigan University to purchase 11,000 tickets at full price in order to accept the invitation to play in the bowl . The university was only able to sell 548
tickets at that price , forcing it to accept a $462,535 loss , before travel expenses , to pay for the privilege of playing in the bowl . Game results ( edit )
Season Date Time ( CST ) Teams Attendance TV 2006 December 28 7 : 00 PM Rutgers 37 Kansas State 10 52,210 NFL Network 2007 December 28 7 : 00 PM TCU 20 Houston
13 62,
</item 5>

<item 6>
Texas Bowl
Kansas State ( Kansas State 33 -- 28 ) 2017 matchup Missouri vs Texas ( Texas 33 -- 16 ) The Texas Bowl , officially the Academy Sports + Outdoors Texas Bowl
for sponsorship purposes , is a post-season NCAA - sanctioned Division I FBS college football bowl game first held in 2006 in Houston , Texas . The bowl
replaced the defunct Houston Bowl , which played annually from 2000 to 2005 , and before that the Bluebonnet Bowl , the first bowl game in Houston , played from
1959 through 1987 . From 2011 to 2012 , the game was sponsored by Meineke Car Care and officially known as the Meineke Car Care Bowl of Texas . The game was
then officially the AdvoCare V100 Texas Bowl from 2014 to 2016 , with AdvoCare as the title sponsor . Since 2017 , the game has been sponsored by Academy Sports
+ Outdoors .
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
Rank Conference Appearances Wins Losses Pct . Big 12 9 5 . 444 T2 SEC . 500 T2 Big Ten . 250 C - USA . 500 T5 ACC 0 1.000 T5 Big East 0 1.000 T5 Independents 0
1.000 T5 Mountain West 0 1.000 T5 MAC 0 . 000 Jump up ^ Rutgers ( 2006 ) appeared as a member of the Big East . Following the 2013 split of the original Big
East along football lines , the FBS schools reorganized as the new American Athletic Conference , which retains the charter of the original Big East . Jump up ^
Navy ( 2009 ) See also ( edit ) List of college bowl games List of Texas Bowl broadcasters References ( edit ) Jump up ^ `` College Bowl Game Payouts '' .
statisticbrain.com. 2015 . Jump up ^ Duarte , Joseph ( 18 April 2006 ) . `` Houston Bowl in jeopardy '' . Houston Chronicle . Retrieved 29 December 2015 . Jump
up ^ `` NFL Network gets bowl game in Houston '' .
</item 8>

<item 9>
Texas Bowl
It was also announced on December 30 , 2009 , that ESPN Plus would take over as part owner and operator of the game , while Lone Star Sports and Entertainment
will maintain a stake in the bowl , and would be carried on ESPN . Sponsors ( edit ) On April 12 , 2011 , ESPN announced Meineke Car Care signed a three - year
title sponsorship deal beginning in 2011 , changing name of the bowl to the Meineke Car Care Bowl of Texas . On February 12 , 2014 , it was announced that
AdvoCare will be the title sponsor for the bowl game . That sponsorship concluded after the 2016 game . On November 15 , 2017 , Academy Sports + Outdoors became
the new title sponsor of the bowl . Conference tie - ins ( edit ) On May 17 , 2007 , it was announced Conference USA would have a team in the 2007 Texas Bowl .
The Texas Bowl has a rotating commitment with the Big East Conference and Conference USA for 2006 -- 09 while the Big 12 Conference will have a team in all four
of those games .
</item 9>

<item 10>
Texas Bowl
November 15 , 2017 . Jump up ^ Murphy , Austin ; Wetzel , Dan ( 15 November 2010 ) . `` Does It Matter ? '' . Sports Illustrated . p. 47 . External links ( edit
) Official Website Texas Bowl 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 NRG Stadium ( 2006 -- present ) Broadcasters Previous Bowls Bluebonnet Bowl
Houston Bowl NCAA Division I FBS bowl games College Football Playoff Championship Game Cotton Fiesta Orange Peach Rose Sugar Other bowl games Alamo Arizona
Armed Forces Bahamas Belk Birmingham Boca Raton Cactus Camellia Camping World Citrus Cure Dollar General Famous Idaho Potato Foster Farms Frisco Gasparilla
Hawaii Heart of Dallas Holiday Independence Las Vegas Liberty Military Music City New Mexico New Orleans Outback Pinstripe Quick Lane Sun TaxSlayer Texas All -
Star games East -- West Shrine Game NFLPA Collegiate Bowl Senior Bowl Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=d933e8ab223bf087 ===

=== ITEM id=c56493c9dd1845c2 ===
Gold sentences and facts:
Sentence 0: The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet Bandits and must outrun and out - prank them again .
  Fact 0.0: Kevin and his family decide to take a trip to Florida.
  Fact 0.1: Kevin takes the wrong plane.
  Fact 0.2: Kevin ends up in New York City.
  Fact 0.3: Kevin tries to make do with what he has.
  Fact 0.4: Kevin uses his father's credit card to stay at the Plaza Hotel.
  Fact 0.5: Kevin is soon confronted by the Wet Bandits.
  Fact 0.6: Kevin must outrun and out-prank the Wet Bandits again.

Retrieved items:
<item 1>
Home Alone 2: Lost in New York
Peter suddenly calls out , `` Kevin , you spent $967 on room service ? ! '' at which point Kevin runs off . Cast ( edit ) Main cast ( edit ) Macaulay Culkin as
Kevin McCallister Joe Pesci as Harry Lime Daniel Stern as Marv Merchants The McCallister family ( edit ) Catherine O'Hara as Kate McCallister John Heard as
Peter McCallister Devin Ratray as Buzz McCallister , Kevin 's brother who gets him into trouble Kieran Culkin ( Macaulay Culkin 's younger brother ) as Fuller
McCallister Gerry Bamman as Frank McCallister , Kevin 's uncle Hillary Wolf as Megan McCallister Maureen Elisabeth Shay ( who replaces Angela Goethals ) as
Linnie McCallister Mike Maronna as Jeff McCallister Terrie Snell as Leslie McCallister , Kevin 's aunt Jedidiah Cohen as Rod McCallister Senta Moses as Tracy
McCallister Daiana Campeanu as Sondra McCallister Anna Slotky as Brooke McCallister Other characters ( edit ) Brenda Fricker as the Pigeon Lady Tim Curry as Mr.
</item 1>

<item 2>
Home Alone 2: Lost in New York
It is the second film in the Home Alone series and the sequel to Home Alone . Macaulay Culkin reprises his role as Kevin McCallister , while Joe Pesci and
Daniel Stern reprise their roles as the Wet Bandits , now known as the Sticky Bandits . Catherine O'Hara , John Heard , Rob Schneider , Tim Curry , and Brenda
Fricker are also featured . The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New
York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet
Bandits and must outrun and out - prank them again . The film was shot in Winnetka , Illinois ; O'Hare International Airport in Chicago ; Evanston , Illinois ;
and New York City ( which was star Culkin 's hometown at the time ) . The exterior of Duncan 's Toy Chest in New York City was filmed outside of the Rookery
Building in downtown Chicago .
</item 2>

<item 3>
Home Alone 2: Lost in New York
When the duo chase Kevin around the townhouse , he escapes and calls the police . Harry and Marv catch him and take him down to Central Park to kill him , but
the pigeon lady sneaks in and throws birdseed on the duo , and the pigeons storm in to incapacitate them . Kevin sets off fireworks he had bought earlier to
signal the police . The police arrive and arrest the bandits . At the toy store , Mr. Duncan finds a note from Kevin attached to the brick , explaining about
what happened . Kate , still searching for Kevin on foot , remembers his fondness for Christmas trees . After observing Kevin making a wish at the Rockefeller
Center Christmas Tree , Kate meets him there , and they reconcile . On Christmas Day , a truckload of gifts arrive at the McCallisters ' hotel room from the toy
store . Kevin and Buzz reconcile , and Buzz allows him to open up the first present . Kevin goes to Central Park to give the pigeon lady the second turtledove .
At the Plaza , Buzz receives the bill for Kevin 's stay from Cedric and shows it to Peter .
</item 3>

<item 4>
Home Alone 2: Lost in New York
Meanwhile , Kevin goes to his uncle Rob 's townhouse , only to find the house vacant and undergoing renovations , while Rob and his family are in Paris . At
Central Park , he encounters and befriends the pigeon lady . They go to Carnegie Hall , where an orchestra is performing . The pigeon lady explains how her life
collapsed and how she dealt with it by taking care of the pigeons in the park . Kevin gives the pigeon lady some advice and promises that he will be her friend
. Kevin , after remembering what the bandits said , returns to the townhouse , and rigs it with numerous booby traps . Kevin arrives at the toy store during
Harry and Marv 's robbery and throws a brick through the window , setting off the store 's alarm . Kevin then lures the duo to the townhouse , where he springs
the traps and Harry and Marv suffer various injuries . When the duo chase Kevin around the townhouse , he escapes and calls the police .
</item 4>

<item 5>
Home Alone 2: Lost in New York
Duncan . Kevin learns that the proceeds from the store 's Christmas sales will be donated to a children 's hospital . Kevin donates $20 to the hospital . As a
token of appreciation , Mr. Duncan offers Kevin a pair of ceramic turtledoves as a gift , instructing him to give one to another person as a sign of eternal
friendship . After running into Harry and Marv , who have recently escaped from prison and are now called `` The Sticky Bandits '' , Kevin retreats to the Plaza
Hotel . The hotel 's concierge , Mr. Hector , confronts Kevin about the credit card , which has been reported stolen . Kevin flees after evading Mr. Hector ,
but is captured by Harry and Marv . The duo discuss plans for breaking into the toy store that night . Kevin manages to escape by pinching a woman in front of
him , which results in the woman attacking the duo . Kevin 's family travels to New York after tracking the whereabouts of the stolen credit card , and Kate
searches the city for Kevin .
</item 5>

<item 6>
*Home Alone 2: Lost in New York* (1992), directed by Chris Columbus, written/produced by John Hughes, and scored by John Williams, stars Macaulay Culkin as
Kevin, stranded in New York and pursued by the Sticky Bandits. It earned $359 million worldwide. Its soundtrack features Williams’s score and Christmas songs;
expanded editions added cues. Home video releases followed in 2009–15, with a World Trade Center scene later removed. The text also
</item 6>

<item 7>
*Home Alone 2: Lost in New York* is the 1992 sequel starring Macaulay Culkin as Kevin McCallister, with Joe Pesci and Daniel Stern returning as the Sticky
Bandits. After accidentally boarding the wrong plane, Kevin ends up alone in New York, stays at the Plaza Hotel using his father’s credit card, and faces the
burglars again. Filmed in Chicago, Illinois, New York, and Los Angeles, it earned $359 million worldwide on a
</item 7>

<item 8>
Home Alone 2: Lost in New York
In the beginning of the novelization , a prologue , which ends up being Marv 's nightmare in prison , he and Harry sneak away from the cops and return to Kevin
's house to seek revenge on Kevin . Kevin bolts into the garage with Marv and Harry in hot pursuit . Harry and Marv end up triggering extra traps that Kevin had
set up in the garage . Kevin watches as Marv ends up triggering a trap where a running lawnmower falls on his head ( This was a trap featured in Home Alone 3 )
. Sequels ( edit ) Main article : Home Alone 3 A second sequel , Home Alone 3 , followed in 1997 . Two additional television movie sequels , Taking Back the
House and The Holiday Heist , would follow in 2002 and 2012 , respectively . See also ( edit ) List of films featuring fictional films References ( edit ) Jump
up ^ `` HOME ALONE 2 - LOST IN NEW YORK ( PG ) '' . British Board of Film Classification . November 4 , 2017 . Retrieved November 10 , 2015 .
</item 8>

<item 9>
Home Alone 2: Lost in New York
Contents ( hide ) 1 Plot 2 Cast 2.1 Main cast 2.2 The McCallister family 2.3 Other characters 3 Release 3.1 Marketing 3.2 Box office 3.3 Critical reception 3.4
Home media 3.5 Social Media 4 Music 4.1 Original Score 4.2 Original Soundtrack Album 4.3 The Deluxe Edition 5 Novelization 6 Sequels 7 See also 8 References 9
External links Plot ( edit ) In Chicago , the McCallister family is preparing for a Christmas vacation in Miami . On the night before their departure , the
entire family gathers at Peter and Kate 's home , where their youngest son , Kevin , sees Florida as contradictory to Christmas , on his opinion of the lack of
Christmas trees in Florida . During the school Christmas pageant , Kevin 's older brother Buzz humiliates him during his solo , causing Kevin to retaliate ,
thereby ruining the performance .
</item 9>

<item 10>
Home Alone 2: Lost in New York
While Buzz fakes an apology to Kevin in front of the McCallisters ( who are convinced it is sincere ) , Kevin refuses to apologize to Buzz and storms to the
third floor of the house . Prior to the pageant , Peter has unknowingly caused the alarm clock to reset ; consequently , the family once again oversleeps . In
the confusion they have to rush to reach the airport on time , Kevin falls behind everyone else , and mistakenly boards a flight bound for New York City while
trying to replace the batteries for a Talkboy , carrying Peter 's bag containing his wallet and a large amount of money ; upon arrival in Miami , Kate realizes
that Kevin is missing again . In New York , Kevin tours the city and meets a homeless woman tending to pigeons at Central Park , where he runs off . Kevin goes
to the Plaza Hotel , where he tantalizes the staff into checking in with Peter 's credit card . On Christmas Eve , Kevin tours the city in a limousine and
visits a toy store where he meets its philanthropic owner , Mr. Duncan .
</item 10>
=== END ITEM id=c56493c9dd1845c2 ===

=== ITEM id=e18e7dddd26f559d ===
Gold sentences and facts:
Sentence 0: Late in the 20th century , lithium became an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge - and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5 volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries , which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium - ion polymer battery , lithium iron phosphate battery , and the nanowire battery .
  Fact 0.0: Late in the 20th century, lithium became an important component of battery electrolytes.
  Fact 0.1: Late in the 20th century, lithium became an important component of battery electrodes.
  Fact 0.2: Lithium became an important battery component because of its high electrode potential.
  Fact 0.3: Lithium has a low atomic mass.
  Fact 0.4: Because of its low atomic mass, lithium has a high charge-to-weight ratio.
  Fact 0.5: Because of its low atomic mass, lithium has a high power-to-weight ratio.
  Fact 0.6: A typical lithium-ion battery can generate approximately 3 volts per cell.
  Fact 0.7: Lead-acid batteries generate 2.1 volts.
  Fact 0.8: Zinc-carbon cells generate 1.5 volts.
  Fact 0.9: Lithium-ion batteries are rechargeable.
  Fact 0.10: Lithium-ion batteries have a high energy density.
  Fact 0.11: Lithium-ion batteries should not be confused with lithium batteries.
  Fact 0.12: Lithium batteries are disposable (primary) batteries.
  Fact 0.13: Lithium batteries have lithium or its compounds as the anode.
  Fact 0.14: Other rechargeable batteries that use lithium include the lithium-ion polymer battery.
  Fact 0.15: Other rechargeable batteries that use lithium include the lithium iron phosphate battery.
  Fact 0.16: Other rechargeable batteries that use lithium include the nanowire battery.
Sentence 1: Lithium and its compounds have several industrial applications , including heat - resistant glass and ceramics , lithium grease lubricants , flux additives for iron , steel and aluminium production , lithium batteries , and lithium - ion batteries . These uses consume more than three quarters of lithium production .
  Fact 1.0: Lithium and its compounds have several industrial applications.
  Fact 1.1: Lithium and its compounds are used in heat-resistant glass and ceramics.
  Fact 1.2: Lithium and its compounds are used in lithium grease lubricants.
  Fact 1.3: Lithium and its compounds are used as flux additives for iron, steel and aluminium production.
  Fact 1.4: Lithium and its compounds are used in lithium batteries.
  Fact 1.5: Lithium and its compounds are used in lithium-ion batteries.
  Fact 1.6: These uses consume more than three quarters of lithium production.

Retrieved items:
<item 1>
Lithium
Each batch takes from 18 to 24 months . In 1998 , the price of lithium was about 95 USD / kg ( or 43 USD / lb ) . Reserves ( edit ) Worldwide identified
reserves in 2008 were estimated by the US Geological Survey ( USGS ) to be 13 million tonnes , though an accurate estimate of world lithium reserves is
difficult . One reason for this is that most lithium classification schemes are developed for solid ore deposits , whereas brine is a fluid that is problematic
to treat with the same classification scheme due to varying concentrations and pumping effects . The world has been estimated to contain about 15 million tonnes
of lithium reserves , while 65 million tonnes of known resources are reasonable . A total of 75 % of everything can typically be found in the ten largest
deposits of the world . Another study noted that 83 % of the geological resources of lithium are located in six brine , two pegmatite , and two sedimentary
deposits . Deposits are found in South America throughout the Andes mountain chain . Chile is the leading producer , followed by Argentina . Both countries
recover lithium from brine pools .
</item 1>

<item 2>
Lithium
The stockpiled lithium was depleted in lithium - 6 by 75 % , which was enough to affect the measured atomic weight of lithium in many standardized chemicals ,
and even the atomic weight of lithium in some `` natural sources '' of lithium ion which had been `` contaminated '' by lithium salts discharged from isotope
separation facilities , which had found its way into ground water . Lithium was used to decrease the melting temperature of glass and to improve the melting
behavior of aluminium oxide when using the Hall - Héroult process . These two uses dominated the market until the middle of the 1990s . After the end of the
nuclear arms race , the demand for lithium decreased and the sale of Department of Energy stockpiles on the open market further reduced prices . In the mid
1990s , several companies started to extract lithium from brine which proved to be a less expensive than underground or open - pit mining . Most of the mines
closed or shifted their focus to other materials because only the ore from zoned pegmatites could be mined for a competitive price . For example , the US mines
near Kings Mountain , North Carolina closed before the beginning of the 21st century .
</item 2>

<item 3>
Lithium
000,000 Portugal 200 60,000 N / A Russia - - 1,000,000 Serbia - - 1,000,000 United States 38,000 6,900,000 Zimbabwe 900 23,000 100,000 + World total 32,500
14,000,000 46,900,000 See also : Lithium minerals Although lithium is widely distributed on Earth , it does not naturally occur in elemental form due to its
high reactivity . The total lithium content of seawater is very large and is estimated as 230 billion tonnes , where the element exists at a relatively constant
concentration of 0.14 to 0.25 parts per million ( ppm ) , or 25 micromolar ; higher concentrations approaching 7 ppm are found near hydrothermal vents .
Estimates for the Earth 's crustal content range from 20 to 70 ppm by weight . In keeping with its name , lithium forms a minor part of igneous rocks , with the
largest concentrations in granites . Granitic pegmatites also provide the greatest abundance of lithium - containing minerals , with spodumene and petalite
being the most commercially viable sources .
</item 3>

<item 4>
Lithium
Extraction ( edit ) Lithium salts are extracted from water in mineral springs , brine pools , and brine deposits . Lithium is present in seawater , but
commercially viable methods of extraction have yet to be developed . Another potential source of lithium is the leachates of geothermal wells , which are
carried to the surface . Recovery of lithium has been demonstrated in the field ; the lithium is separated by simple filtration . The process and environmental
costs are primarily those of the already - operating well ; net environmental impacts may thus be positive . Uses ( edit ) Estimates of global lithium uses in
2015 Ceramics and glass ( 32 % ) Batteries ( 35 % ) Lubricating greases ( 9 % ) Continuous casting ( 5 % ) Air treatment ( 5 % ) Polymers ( 4 % ) Primary
aluminum production ( 1 % ) Pharmaceuticals ( < 1 % ) Other ( 9 % ) Ceramics and glass ( edit ) Lithium oxide is widely used as a flux for processing silica ,
reducing the melting point and viscosity of the material and leading to glazes with improved physical properties including low coefficients of thermal expansion
.
</item 4>

<item 5>
Lithium
We are not aware of any discoveries of lithium , ' he said . '' Lithia ( `` lithium brine '' ) is associated with tin mining areas in Cornwall , England and an
evaluation project from 400 - metre deep test boreholes is under consideration . If successful the hot brines will also provide geothermal energy to power the
lithium extraction and refining process . Biological ( edit ) Lithium is found in trace amount in numerous plants , plankton , and invertebrates , at
concentrations of 69 to 5,760 parts per billion ( ppb ) . In vertebrates the concentration is slightly lower , and nearly all vertebrate tissue and body fluids
contain lithium ranging from 21 to 763 ppb . Marine organisms tend to bioaccumulate lithium more than terrestrial organisms . Whether lithium has a
physiological role in any of these organisms is unknown . History ( edit ) Johan August Arfwedson is credited with the discovery of lithium in 1817 Petalite (
LiAlSi O ) was discovered in 1800 by the Brazilian chemist and statesman José Bonifácio de Andrada e Silva in a mine on the island of Utö , Sweden .
</item 5>

<item 6>
Lithium
Worldwide , this is one of the largest use for lithium compounds . Glazes containing lithium oxides are used for ovenware . Lithium carbonate ( Li CO ) is
generally used in this application because it converts to the oxide upon heating . Electrical and electronics ( edit ) Late in the 20th century , lithium became
an important component of battery electrolytes and electrodes , because of its high electrode potential . Because of its low atomic mass , it has a high charge
- and power - to - weight ratio . A typical lithium - ion battery can generate approximately 3 volts per cell , compared with 2.1 volts for lead - acid or 1.5
volts for zinc - carbon cells . Lithium - ion batteries , which are rechargeable and have a high energy density , should not be confused with lithium batteries
, which are disposable ( primary ) batteries with lithium or its compounds as the anode . Other rechargeable batteries that use lithium include the lithium -
ion polymer battery , lithium iron phosphate battery , and the nanowire battery . Lubricating greases ( edit ) Main article : Lithium grease The third most
common use of lithium is in greases .
</item 6>

<item 7>
Lithium
Another significant mineral of lithium is lepidolite . A newer source for lithium is hectorite clay , the only active development of which is through the
Western Lithium Corporation in the United States . At 20 mg lithium per kg of Earth 's crust , lithium is the 25th most abundant element . According to the
Handbook of Lithium and Natural Calcium , `` Lithium is a comparatively rare element , although it is found in many rocks and some brines , but always in very
low concentrations . There are a fairly large number of both lithium mineral and brine deposits but only comparatively few of them are of actual or potential
commercial value . Many are very small , others are too low in grade . '' The US Geological Survey estimates that in 2010 , Chile had the largest reserves by
far ( 7.5 million tonnes ) and the highest annual production ( 8,800 tonnes ) . One of the largest reserve bases of lithium is in the Salar de Uyuni area of
Bolivia , which has 5.4 million tonnes . Other major suppliers include Australia , Argentina and China .
</item 7>

<item 8>
Lithium
Lithium hydroxide is a strong base and , when heated with a fat , produces a soap made of lithium stearate . Lithium soap has the ability to thicken oils , and
it is used to manufacture all - purpose , high - temperature lubricating greases . Metallurgy ( edit ) Lithium ( e.g. as lithium carbonate ) is used as an
additive to continuous casting mould flux slags where it increases fluidity , a use which accounts for 5 % of global lithium use ( 2011 ) . Lithium compounds
are also used as additives ( fluxes ) to foundry sand for iron casting to reduce veining . Lithium ( as lithium fluoride ) is used as an additive to aluminium
smelters ( Hall -- Héroult process ) , reducing melting temperature and increasing electrical resistance , a use which accounts for 3 % of production ( 2011 ) .
When used as a flux for welding or soldering , metallic lithium promotes the fusing of metals during the process and eliminates the forming of oxides by
absorbing impurities .
</item 8>

<item 9>
Lithium
When cut open , it exhibits a metallic luster , but moist air corrodes it quickly to a dull silvery gray , then black tarnish . It never occurs freely in nature
, but only in ( usually ionic ) compounds , such as pegmatitic minerals which were once the main source of lithium . Due to its solubility as an ion , it is
present in ocean water and is commonly obtained from brines . Lithium metal is isolated electrolytically from a mixture of lithium chloride and potassium
chloride . The nucleus of the lithium atom verges on instability , since the two stable lithium isotopes found in nature have among the lowest binding energies
per nucleon of all stable nuclides . Because of its relative nuclear instability , lithium is less common in the solar system than 25 of the first 32 chemical
elements even though its nuclei are very light : it is an exception to the trend that heavier nuclei are less common . For related reasons , lithium has
important uses in nuclear physics . The transmutation of lithium atoms to helium in 1932 was the first fully man - made nuclear reaction , and lithium deuteride
serves as a fusion fuel in staged thermonuclear weapons .
</item 9>

<item 10>
Lithium
Lithium - wikipedia Lithium Jump to : navigation , search This article is about the chemical element . For the use of lithium as a medication , see Lithium (
medication ) . For other uses , see Lithium ( disambiguation ) . It has been suggested that Lithium as an investment be merged into this article . ( Discuss )
Proposed since June 2017 . Lithium , Li Lithium floating in oil General properties Pronunciation / ˈlɪθiəm / ​ ( LI - thee - əm ) Appearance silvery - white
Standard atomic weight ( A ) ( 7000693800000000000 ♠ 6.938 , 7000699700000000000 ♠ 6.997 ) conventional : 7000694000000000000 ♠ 6.94 Lithium in the periodic
table Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium
Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium
</item 10>
=== END ITEM id=e18e7dddd26f559d ===

=== ITEM id=7b2861ec5c013b62 ===
Gold sentences and facts:
Sentence 0: Human fertilization is the union of a human egg and sperm , usually occurring in the ampulla of the fallopian tube . The result of this union is the production of a zygote cell , or fertilized egg , initiating prenatal development . Scientists discovered the dynamics of human fertilization in the nineteenth century .
  Fact 0.0: Human fertilization is the union of a human egg and sperm.
  Fact 0.1: Human fertilization usually occurs in the ampulla of the fallopian tube.
  Fact 0.2: The result of the union of a human egg and sperm is a zygote cell, also called a fertilized egg.
  Fact 0.3: The union of a human egg and sperm initiates prenatal development.
  Fact 0.4: Scientists discovered the dynamics of human fertilization in the nineteenth century.
Sentence 1: The process of fertilization involves a sperm fusing with an ovum . The most common sequence begins with ejaculation during copulation , follows with ovulation , and finishes with fertilization . Various exceptions to this sequence are possible , including artificial insemination , in vitro fertilization , external ejaculation without copulation , or copulation shortly after ovulation . Upon encountering the secondary oocyte , the acrosome of the sperm produces enzymes which allow it to burrow through the outer jelly coat of the egg . The sperm plasma then fuses with the egg 's plasma membrane , the sperm head disconnects from its flagellum and the egg travels down the Fallopian tube to reach the uterus .
  Fact 1.0: The process of fertilization involves a sperm fusing with an ovum.
  Fact 1.1: The most common sequence of fertilization begins with ejaculation during copulation.
  Fact 1.2: The most common sequence of fertilization is followed by ovulation.
  Fact 1.3: The most common sequence of fertilization finishes with fertilization.
  Fact 1.4: Various exceptions to the most common sequence of fertilization are possible.
  Fact 1.5: One exception to the sequence of fertilization is artificial insemination.
  Fact 1.6: One exception to the sequence of fertilization is in vitro fertilization.
  Fact 1.7: One exception to the sequence of fertilization is external ejaculation without copulation.
  Fact 1.8: One exception to the sequence of fertilization is copulation shortly after ovulation.
  Fact 1.9: Upon encountering the secondary oocyte, the acrosome of the sperm produces enzymes.
  Fact 1.10: The enzymes from the acrosome of the sperm allow the sperm to burrow through the outer jelly coat of the egg.
  Fact 1.11: The sperm plasma fuses with the egg's plasma membrane.
  Fact 1.12: The sperm head disconnects from its flagellum.
  Fact 1.13: The egg travels down the Fallopian tube to reach the uterus.

Retrieved items:
<item 1>
Human fertilization
1 Cortical reaction 2 Fusion 2.1 Cell membranes 2.2 Transformations 2.3 Replication 2.4 Mitosis 3 Fertilization age 4 Diseases 5 See also 6 References 7
External links Anatomy ( edit ) Corona radiata ( edit ) The sperm binds through the corona radiata , a layer of follicle cells on the outside of the secondary
oocyte . Fertilization occurs when the nucleus of both a sperm and an egg fuse to form a diploid cell , known as zygote . The successful fusion of gametes forms
a new organism . Cone of attraction and perivitelline membrane ( edit ) Where the spermatozoon is about to pierce , the yolk ( ooplasm ) is drawn out into a
conical elevation , termed the cone of attraction or reception cone . Once the spermatozoon has entered , the peripheral portion of the yolk changes into a
membrane , the perivitelline membrane , which prevents the passage of additional spermatozoa .
</item 1>

<item 2>
Human fertilization
Human fertilization - wikipedia Human fertilization Jump to : navigation , search Part of a series on Human growth and development Stages Human embryogenesis
Fetus Infant Toddler Early childhood Child Preadolescence Adolescence Adult Middle age Old age Biological milestones Fertilization Childbirth Walking Language
acquisition Puberty Menopause Ageing Death Development and psychology Pre - and perinatal Infant and child Adolescent Youth Young adult Adult Maturity
Developmental stage theories Attachment Ecological Psychosocial Psychosexual development Moral Cognitive Cultural - historical Evolutionary Human body portal
The acrosome reaction for a sea urchin , a similar process . Note that the picture shows several stages of one and the same spermatozoon - only one penetrates
the ovum Illustration depicting ovulation and fertilization . Human fertilization is the union of a human egg and sperm , usually occurring in the ampulla of
the fallopian tube . The result of this union is the production of a zygote cell , or fertilized egg , initiating prenatal development . Scientists discovered
the dynamics of human fertilization in the nineteenth century . The process of fertilization involves a sperm fusing with an ovum .
</item 2>

<item 3>
Human fertilization
Each of the two daughter cells resulting from that mitosis has one replica of each chromatid that was replicated in the previous stage . Thus , they are
genetically identical . Fertilization age ( edit ) Fertilization is the event most commonly used to mark the zero point in descriptions of prenatal development
of the embryo or fetus . The resultant age is known as fertilization age , fertilizational age , embryonic age , fetal age or ( intrauterine ) developmental (
IUD ) age . Gestational age , in contrast , takes the beginning of the last menstrual period ( LMP ) as the zero point . By convention , gestational age is
calculated by adding 14 days to fertilization age and vice versa . In fact , however , fertilization usually occurs within a day of ovulation , which , in turn
, occurs on average 14.6 days after the beginning of the preceding menstruation ( LMP ) . There is also considerable variability in this interval , with a 95 %
prediction interval of the ovulation of 9 to 20 days after menstruation even for an average woman who has a mean LMP - to - ovulation time of 14.6 .
</item 3>

<item 4>
Human fertilization
As a population , sperm cells have on average 50 % genome similarity so the premature acrosomal reactions aid fertilization by a member of the same cohort . It
may be regarded as a mechanism of kin selection . Recent studies have shown that the egg is not passive during this process . Cortical reaction ( edit ) Once
the sperm cells find their way past the zona pellucida , the cortical reaction occurs . Cortical granules inside the secondary oocyte fuse with the plasma
membrane of the cell , causing enzymes inside these granules to be expelled by exocytosis to the zona pellucida . This in turn causes the glyco - proteins in
the zona pellucida to cross-link with each other -- i.e. the enzymes cause the ZP2 to hydrolyse into ZP2f -- making the whole matrix hard and impermeable to
sperm . This prevents fertilization of an egg by more than one sperm . The cortical reaction and acrosome reaction are both essential to ensure that only one
sperm will fertilize an egg . Fusion ( edit ) Fertilization and implantation in humans .
</item 4>

<item 5>
Gamete
Gamete - wikipedia Gamete Jump to : navigation , search Part of a series on Sex Biological terms Sexual dimorphism Sexual differentiation Feminization
Virilization Sex - determination system XY X0 ZW Z0 Temperature - dependent Haplodiploidy Heterogametic sex Homogametic sex Sex chromosome X chromosome Y
chromosome Testis - determining factor Hermaphrodite Sequential hermaphroditism Intersex Sexual reproduction Evolution of sexual reproduction Anisogamy Isogamy
Germ cell Meiosis Gametogenesis Spermatogenesis Oogenesis Gamete spermatozoon ovum Fertilization External fertilization Internal fertilization Sexual selection
Plant reproduction Fungal reproduction Sexual reproduction in animals Sexual intercourse Copulation Human reproduction Sexuality Plant sexuality Animal
sexuality Human sexuality Mechanics Differentiation Activity Sex portal Biology portal A gamete ( from Ancient Greek γαμετή gamete from gamein `` to marry '' )
is a haploid cell that fuses with another haploid cell during fertilization ( conception ) in organisms that sexually reproduce .
</item 5>

<item 6>
Human fertilization
Fusion ( edit ) Fertilization and implantation in humans . After the sperm enters the cytoplasm of the oocyte ( also called ovocyte ) , the tail and the outer
coating of the sperm disintegrate and the cortical reaction takes place , preventing other sperm from fertilizing the same egg . The oocyte now undergoes its
second meiotic division producing the haploid ovum and releasing a polar body . The sperm nucleus then fuses with the ovum , enabling fusion of their genetic
material . Cell membranes ( edit ) The fusion of cell membranes of the secondary oocyte and sperm takes place . Transformations ( edit ) In preparation for the
fusion of their genetic material both the oocyte and the sperm undergo transformations as a reaction to the fusion of cell membranes . The oocyte completes its
second meiotic division . This results in a mature ovum . The nucleus of the oocyte is called a pronucleus in this process , to distinguish it from the nuclei
that are the result of fertilization . The sperm 's tail and mitochondria degenerate with the formation of the male pronucleus . This is why all mitochondria in
humans are of maternal origin .
</item 6>

<item 7>
Human fertilization
The process of fertilization involves a sperm fusing with an ovum . The most common sequence begins with ejaculation during copulation , follows with ovulation
, and finishes with fertilization . Various exceptions to this sequence are possible , including artificial insemination , in vitro fertilization , external
ejaculation without copulation , or copulation shortly after ovulation . Upon encountering the secondary oocyte , the acrosome of the sperm produces enzymes
which allow it to burrow through the outer jelly coat of the egg . The sperm plasma then fuses with the egg 's plasma membrane , the sperm head disconnects from
its flagellum and the egg travels down the Fallopian tube to reach the uterus . In vitro fertilization ( IVF ) is a process by which egg cells are fertilized by
sperm outside the womb , in vitro . Contents ( hide ) 1 Anatomy 1.1 Corona radiata 1.2 Cone of attraction and perivitelline membrane 1.3 Sperm preparation 1.4
Zona pellucida 1.4.
</item 7>

<item 8>
Human fertilization
This is why all mitochondria in humans are of maternal origin . Still , a considerable amount of RNA from the sperm is delivered to the resulting embryo and
likely influences embryo development and the phenotype of the offspring . Replication ( edit ) The pronuclei migrate toward the center of the oocyte , rapidly
replicating their DNA as they do so to prepare the zygote for its first mitotic division . Mitosis ( edit ) Usually 23 chromosomes from spermatozoon and 23
chromosomes from egg cell fuse ( half of spermatozoons carry X chromosome and the other half Y chromosome ) . Their membranes dissolve , leaving no barriers
between the male and female chromosomes . During this dissolution , a mitotic spindle forms between them . The spindle captures the chromosomes before they
disperse in the egg cytoplasm . Upon subsequently undergoing mitosis ( which includes pulling of chromatids towards centrioles in anaphase ) the cell gathers
genetic material from the male and female together . Thus , the first mitosis of the union of sperm and oocyte is the actual fusion of their chromosomes .
</item 8>

<item 9>
Human fertilization
Jump up ^ `` Fertilization '' . Retrieved 28 July 2010 . Jump up ^ Angier , Natalie ( 2007 - 06 - 12 ) . `` Sleek , Fast and Focused : The Cells That Make Dad
Dad '' . The New York Times . Jump up ^ Suzanne Wymelenberg , Science and Babies , National Academy Press , page 17 Jump up ^ Richard E. Jones and Kristin H.
Lopez , Human Reproductive Biology , Third Edition , Elsevier , 2006 , page 238 Jump up ^ `` Fertilization : The Cortical Reaction '' . Boundless . Boundless .
Retrieved 14 March 2013 . Jump up ^ Jodar , M. ; Selvaraju , S. ; Sendler , E. ; Diamond , M.P. ; Krawetz , S.A. ; for the Reproductive Medicine Networks ( 2013
) . `` The presence , role and clinical use of spermatozoal RNAs '' . Human Reproduction Update. 19 ( 6 ) : 604 -- 624 .
</item 9>

<item 10>
Human fertilization
Sperm preparation ( edit ) Further information : Acrosome reaction At the beginning of the process , the sperm undergoes a series of changes , as freshly
ejaculated sperm is unable or poorly able to fertilize . The sperm must undergo capacitation in the female 's reproductive tract over several hours , which
increases its motility and destabilizes its membrane , preparing it for the acrosome reaction , the enzymatic penetration of the egg 's tough membrane , the
zona pellucida , which surrounds the oocyte . Zona pellucida ( edit ) After binding to the corona radiata the sperm reaches the zona pellucida , which is an
extra-cellular matrix of glycoproteins . A special complementary molecule on the surface of the sperm head binds to a ZP3 glycoprotein in the zona pellucida .
This binding triggers the acrosome to burst , releasing enzymes that help the sperm get through the zona pellucida . Some sperm cells consume their acrosome
prematurely on the surface of the egg cell , facilitating the penetration by other sperm cells .
</item 10>
=== END ITEM id=7b2861ec5c013b62 ===

=== ITEM id=6f875f334806d28d ===
Gold sentences and facts:
Sentence 0: `` Bad Boys '' is a 1987 song by the Jamaican reggae band Inner Circle , which obtained high popularity in the United States after its re-release in 1993 , peaking at number 8 on the Billboard Hot 100 and number 7 on the Top 40 Mainstream . It is well known as the opening theme to the U.S. TV show Cops .
  Fact 0.0: Bad Boys is a 1987 song.
  Fact 0.1: Bad Boys is a song by the Jamaican reggae band Inner Circle.
  Fact 0.2: Bad Boys obtained high popularity in the United States after its re-release in 1993.
  Fact 0.3: Bad Boys peaked at number 8 on the Billboard Hot 100.
  Fact 0.4: Bad Boys peaked at number 7 on the Top 40 Mainstream.
  Fact 0.5: Bad Boys is well known as the opening theme to the U.S. TV show Cops.

Retrieved items:
<item 1>
Jack McCoy
( hide ) Law & Order Characters Police Sgt . Max Greevey Sgt . Phil Cerreta Lennie Briscoe Joe Fontana Ed Green Cyrus Lupo Mike Logan Rey Curtis Nick Falco Nina
Cassady Kevin Bernard Captain Donald Cragen Lieutenant Anita Van Buren District attorneys Benjamin Stone Jack McCoy Michael Cutter Paul Robinette Claire Kincaid
Jamie Ross Abbie Carmichael Serena Southerlyn Alexandra Borgia Connie Rubirosa Adam Schiff Nora Lewin Arthur Branch Recurring characters Elizabeth Olivet , Ph.
D . Emil Skoda , M.D. Elizabeth Rodgers , M.D.
</item 1>

<item 2>
Jack McCoy
Episode 8 . November 29 , 1995 . NBC . Jump up ^ `` Thrill '' . Law & Order . Season 8 . Episode 1 . September 24 , 1997 . NBC . Jump up ^ `` Under God '' . Law
& Order . Season 13 . Episode 12 . February 5 , 2003 . NBC . Jump up ^ `` Listing for `` Confidential '' `` . The Futon Critic ( Press release ) . NBC Universal
. February 19 , 2010 . ^ Jump up to : Tucker , Ken . `` TV : Sam Waterston 's bark keeps giving Law & Order its bite '' , Entertainment Weekly 1026 ( December
19 , 2008 ) : 49 . ( hide ) Law & Order Characters Police Sgt . Max Greevey Sgt .
</item 2>

<item 3>
Bad Boys (Inner Circle song)
Chart performance ( edit ) Weekly charts ( edit ) Year Chart Peak position 1991 Finland ( Suomen virallinen lista ) Norway ( VG - lista ) Sweden (
Sverigetopplistan ) Australia ( ARIA ) 25 Austria ( Ö3 Austria Top 40 ) 21 Canada Top Singles ( RPM ) 19 Germany ( Official German Charts ) 35 Ireland ( IRMA )
26 New Zealand ( Recorded Music NZ ) 5 UK Singles ( Official Charts Company ) 52 US Billboard Hot 100 8 US Billboard Hot Dance Music / Maxi - Singles Sales 15
US Billboard Hot R&B Singles 58 US Billboard Rhythmic Top 40 27 US Billboard Top 40 Mainstream 7 2006 US Billboard Hot Ringtones 10 2007 US Billboard Hot
Ringtones 20 Year - end charts ( edit ) Chart ( 1993 ) Rank New Zealand ( Recorded Music NZ ) 15 US Billboard Hot 100 69 In popular culture ( edit ) `` Bad Boys
'' was selected as the theme song to Cops because a field producer for the show happened to be a fan of Inner Circle .
</item 3>

<item 4>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 4>

<item 5>
Jack McCoy
It 's You '' Season 7 Episode 15 : `` Sideshow '' Law & Order : Special Victims Unit Season 1 Episode 15 : `` Entitled '' Season 9 Episode 7 : `` Blinded ''
Season 11 Episode 21 : `` Torch '' Season 19 Episode 13 : `` The Undiscovered Country '' Law & Order : Trial by Jury Episode 1 : `` The Abominable Showman ''
Episode 8 : `` Skeleton '' References ( edit ) Jump up ^ ( 1 ) . New York Landmarks Conservancy . ^ Jump up to : `` Trophy '' . Law & Order . Season 6 . Episode
12 . January 31 , 1996 . NBC . Jump up ^ `` Chosen '' . Law & Order . Season 13 . Episode 11 . January 15 , 2003 . NBC . Jump up ^ `` The Drowned and the Saved
'' . Law & Order . Season 19 . Episode 22 . June 3 , 2009 . NBC . ^ Jump up to : `` Aftershock '' . Law & Order . Season 6 .
</item 5>

<item 6>
Bad Boys (Inner Circle song)
It is well known as the opening theme to the U.S. TV show Cops . Contents ( hide ) 1 History 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 In
popular culture 4 See also 5 References 6 External links History ( edit ) The song was originally released in 1987 on the album One Way and obtained minor
airplay and poor sales . However , in 1992 , it was also included on the Bad to the Bone album , and the song was released as a single in 1993 , after the
unexpected success of their previous international hit single `` Sweat ( A La La La La Long ) '' . `` Bad Boys '' reached number 52 in the United Kingdom and
number 8 in the United States ; in the latter territory , it was certified Gold and sold 600,000 copies . The song was also released as B - side on Inner Circle
's 1992 single `` Sweat ( A La La La La Long ) '' . The song was also released as a single in mid-1991 in Norway ; it peaked at number one for three weeks .
</item 6>

<item 7>
Law & Order: Special Victims Unit
Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda
Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13
14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' `` Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice
Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion '' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion
'' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and nominations Law & Order franchise Chicago franchise Chicago Fire
Characters Matthew Casey Kelly Severide Episodes Season 1 5 6 Chicago P.D.
</item 7>

<item 8>
Blue Bloods (season 5)
Danny hears a waitress who sounds like one of the robbers . He later visits the woman 's husband and find he has no legs and is also suffering from traumatic
brain injury the result of combat . He tells the detectives he has had numerous delays in approval for treatment from the Veteran 's Administration , but that
his wife was able to come up with $100,000 in a `` fundraiser '' . Danny and Maria realize the bank robbery was the `` fundraiser '' , but they do not let on
the soldier . Danny also finds out the woman was cheating on her husband , he gets photographic proof and he uses it as leverage to get her to confess . Each
member of the group ultimately receives five years in prison , while Danny and Frank are able to pull some strings with the VA to get the soldier treated
immediately . Meanwhile , the work of a guerilla street artist named Spanky ( in the vein of Banksy ) is causing public scares in the form of pop - ups that
emerge from unattended bags . This creates a rift between Garrett and Gormley on how to treat these actions while maintaining the image of the NYPD .
</item 8>

<item 9>
Blue Bloods (season 5)
Frank tells the married Senator that the arresting officer proclaimed `` off the record '' that the two were having a lovers ' quarrel in the car . McCreary
pleads for a way to bury the arrest and save his reputation , but Frank is more worried about giving the arresting officer the impression that the P.C. can be
bought . 110 21 `` New Rules '' John Behring Ian Biederman April 24 , 2015 ( 2015 - 04 - 24 ) 521 9.99 Shortly after Frank awards Deputy Chief Donald Kent (
Dennis Haysbert ) , commander of the NYPD gang division , with his second star , Kent and his wife Margret are shot and killed in a likely gang hit . Frank
personally assigns Danny to the case , but impresses upon his son that everything must be done by the book . Danny believes he can get information from Hector ,
who witnessed the hit and is in the hospital with a stray bullet wound , but Hector fears the gangs and does n't want to cause any trouble .
</item 9>

<item 10>
The excerpt lists *Law & Order: SVU* cast members, characters, ranks, and season appearances. It includes Christopher Meloni as Detective Elliot Stabler
(seasons 1–12), Mariska Hargitay as Lieutenant Olivia Benson, Richard Belzer as DA Investigator John Munch, Dann Florek as Captain Donald Cragen, Michelle Hurd
as Detective Monique Jeffries, Stephanie March as ADA Alexandra Cabot, Ice-T as Detective Fin Tutuola, and Tam
</item 10>
=== END ITEM id=6f875f334806d28d ===

=== ITEM id=54f6ec6f90331457 ===
Gold sentences and facts:
Sentence 0: The total number of languages natively spoken in Africa is variously estimated ( depending on the delineation of language vs. dialect ) at between 1,250 to 2,100 , and by some counts at `` over 3,000 '' , Nigeria alone has over 500 languages ( according to the count of SIL Ethnologue ) , one of the greatest concentrations of linguistic diversity in the world .
  Fact 0.0: The total number of languages natively spoken in Africa is variously estimated at between 1,250 and 2,100, depending on the delineation of language vs. dialect.
  Fact 0.1: Some counts put the number of languages natively spoken in Africa at 'over 3,000'.
  Fact 0.2: Nigeria alone has over 500 languages, according to the count of SIL Ethnologue.
  Fact 0.3: Nigeria is one of the greatest concentrations of linguistic diversity in the world.

Retrieved items:
<item 1>
Languages of Africa
Languages of Africa - wikipedia Languages of Africa Jump to : navigation , search The traditional language families spoken in Africa : Afroasiatic Nilo -
Saharan Niger -- Congo Bantu Khoisan Indo - European Austronesian The languages of Africa are divided into six major language families : Afroasiatic languages
are spread throughout Western Asia , North Africa , the Horn of Africa and parts of the Sahel . Austronesian languages are spoken in Madagascar . Indo -
European languages are spoken in South Africa and Namibia ( Afrikaans , English , German ) and are used as lingua francas in the former colonies of Britain (
English ) , former colonies of France ( French ) , former colonies of Portugal ( Portuguese ) , former colonies of Spain ( Spanish ) and the current Spanish
territories of Ceuta , Melilla and the Canary Islands ( Spanish ) . Khoe languages are concentrated in the Kalahari Desert of Namibia and Botswana . Niger --
Congo languages ( Bantu and non-Bantu ) cover West , Central , Southeast and Southern Africa . Nilo - Saharan languages ( unity debated ) are spoken from
Tanzania to Sudan and from Chad to Mali .
</item 1>

<item 2>
Languages of Africa
700,000 Malawi , Zimbabwe English Indo - European 6,500,000 ( estimated ) see List of territorial entities where English is an official language French Indo -
European 700,330 ( estimated ) see List of territorial entities where French is an official language Fulani Niger -- Congo 25,000,000 Gikuyu Niger -- Congo
6,600,000 Hausa Afroasiatic 34,000,000 Nigeria , Niger Igbo Niger -- Congo 18,000,000 Kinyarwanda Niger -- Congo 9,800,000 Rwanda Kirundi Niger -- Congo
8,800,000 Burundi Kongo Niger -- Congo 5,600,000 recognised national language of Angola Lingala Niger -- Congo 5,500,000 national language of Democratic
Republic of the Congo Luganda Niger - Congo 4,130,000 native language of Uganda Luo Nilo - Saharan ( probable ) 4,200,000 Malagasy Austronesian 18,000,000
Madagascar Mauritian Creole Indo - European 1,135,000 native language of Mauritius Mossi Niger -- Congo 7,600,
</item 2>

<item 3>
Languages of Africa
These are generally thought to be divergent Niger -- Congo , but debate persists . Ubangian , some 70 languages , centered on the languages of the Central
African Republic ; may be Niger -- Congo Khoe , around 10 languages , the primary family of Khoisan languages of Namibia and Botswana Sandawe , an isolate of
Tanzania , possibly related to Khoe Kx'a , a language of Southern Africa Tuu , or Taa - ǃKwi , two surviving languages Hadza , an isolate of Tanzania Bangime ,
a likely isolate of Mali Jalaa , a likely isolate of Nigeria Laal , a possible isolate of Chad Khoisan is a term of convenience covering some 30 languages
spoken by around 300,000 -- 400,000 people . There are five Khoisan families that have not been shown to be related to each other : Khoe , Tuu and Kx'a , which
are found mainly in Namibia and Botswana , as well as Sandawe and Hadza of Tanzania , which are language isolates . A striking feature of Khoisan languages ,
and the reason they are often grouped together , is their use of click consonants .
</item 3>

<item 4>
Languages of Africa
5 Unclassified languages 1.4. 6 Sign languages 2 Language in Africa 2.1 Official Languages 2.2 Cross-border languages 2.3 Language change and planning 2.4
Demographics 3 Linguistic features 3.1 Phonological 3.2 Syntactic 3.3 Semantic 4 Number of speakers 5 See also 5.1 General 5.2 Works 5.3 Classifiers 6 Notes 7
References 8 External links Language groups ( edit ) Clickable map showing the traditional language families , subfamilies and major languages spoken in Africa
Most languages spoken in Africa belong to one of three large language families : Afroasiatic , Nilo - Saharan and Niger -- Congo . Another hundred belong to
small families such as Ubangian ( sometimes grouped within Niger - Congo ) and the various families called Khoisan , or the Indo - European and Austronesian
language families mainly spoken outside Africa ; the presence of the latter two dates to 2,600 and 1,500 years ago , respectively . In addition , the languages
of Africa languages include several unclassified languages and sign languages .
</item 4>

<item 5>
Languages of Africa
There are several other small families and language isolates , as well as languages that have yet to be classified . In addition , Africa has a wide variety of
sign languages , many of which are language isolates ( see below ) . The total number of languages natively spoken in Africa is variously estimated ( depending
on the delineation of language vs. dialect ) at between 1,250 to 2,100 , and by some counts at `` over 3,000 '' , Nigeria alone has over 500 languages (
according to the count of SIL Ethnologue ) , one of the greatest concentrations of linguistic diversity in the world . Around a hundred languages are widely
used for inter-ethnic communication . Arabic , Somali , Berber , Amharic , Oromo , Igbo , Swahili , Hausa , Manding , Fulani and Yoruba are spoken by tens of
millions of people . Twelve dialect clusters ( which may group up to a hundred linguistic varieties ) are spoken by 75 percent , and fifteen by 85 percent , of
Africans as a first or additional language .
</item 5>

<item 6>
Languages of Africa
000,000 Eritrea Tshiluba Niger -- Congo 6,300,000 ( 1991 ) national language of Democratic Republic of the Congo Tswana Niger -- Congo 5,800,000 South Africa ,
Botswana Umbundu Niger -- Congo 6,000,000 recognised national language of Angola Xhosa Niger -- Congo 7,600,000 South Africa , Zimbabwe Yoruba Niger -- Congo
28,000,000 Zulu Niger -- Congo 10,400,000 South Africa See also ( edit ) Africa portal Language portal General ( edit ) Languages of the African Union Writing
systems of Africa Journal of West African Languages Works ( edit ) Polyglotta Africana The Languages of Africa Classifiers ( edit ) Karl Lepsius Wilhelm Bleek
Carl Meinhof Diedrich Westermann Joseph Greenberg Notes ( edit ) Jump up ^ Heine , Bernd ; Heine , Bernd , eds. ( 2000 ) . African Languages : an Introduction .
Cambridge University Press . Jump up ^ Epstein , Edmund L. ; Kole , Robert , eds. ( 1998 ) . The Language of African Literature . Africa World Press .
</item 6>

<item 7>
Languages of Africa
135,000 native language of Mauritius Mossi Niger -- Congo 7,600,000 Recognised regional language in Burkina Faso Ndebele Niger -- Congo 1,090,000 Statuatory
national language in South Africa Northern Sotho Niger -- Congo 4,600,000 South Africa Oromo Afroasiatic 26,000,000 Ethiopia Portuguese Indo - European
13,700,000 ( estimated ) Angola , Cape Verde , Guinea - Bissau , Equatorial Guinea , Mozambique , São Tomé and Príncipe Sesotho Niger -- Congo 5,600,000 Lesotho
, South Africa , Zimbabwe Shona Niger -- Congo 14,200,000 incl . Manyika , Ndau ( 2000 -- 2006 ) Zimbabwe Somali Afroasiatic 16,600,000 Somalia Spanish Indo -
European 2,700,000 Equatorial Guinea Swahili Niger -- Congo 15,000,000 official in Tanzania , Kenya , Uganda , Rwanda national language of Democratic Republic
of Congo Tigrinya Afroasiatic 7,000,000 Eritrea Tshiluba Niger -- Congo 6,300,
</item 7>

<item 8>
Languages of Africa
In addition to the ( likely modest ) impact of borders , there are also cases of dialect levelling ( such as in Igbo and probably many others ) , koinés ( such
as N'Ko and possibly Runyakitara ) and emergence of new dialects ( such as Sheng ) . In some countries , there are official efforts to develop standardized
language versions . There are also many less widely spoken languages that may be considered endangered languages . Demographics ( edit ) Further information :
Demographics of Africa Of the 1 billion Africans ( in 2009 ) , about 17 percent speak an Arabic dialect . About 10 percent speak Swahili , the lingua franca of
Southeast Africa ; about 5 percent speak a Berber dialect ; and about 5 percent speak Hausa , which serves as a lingua franca in much of the Sahel . Other
important West African languages are Yoruba , Igbo and Fula . Major Horn of Africa languages are Amharic , Oromo and Somali . Important South African languages
are Zulu , Xhosa and Afrikaans . English , French and Portuguese are important languages in Africa .
</item 8>

<item 9>
Languages of Africa
Case , tense and other categories may be distinguished only by tone . Semantic ( edit ) Quite often , only one term is used for both animal and meat ; the word
nama or nyama for animal / meat is particularly widespread in otherwise widely divergent African languages . Number of speakers ( edit ) The following is a
table displaying the number of speakers of given languages within Africa : Language Family Native speakers ( L1 ) Official status per country Afrikaans Indo -
European 7,200,000 South Africa Akan Niger -- Congo 11,000,000 None . Government sponsored language of Ghana Amharic Afroasiatic 21,800,000 Ethiopia Arabic
Afroasiatic 150,000,000 but with separate mutually unintelligible varieties Algeria , Chad , Comoros , Djibouti , Egypt , Eritrea , Libya , Mauritania , Morocco
, Somalia , Sudan , Tunisia Berber Afroasiatic 56,000,000 ( estimated ) ( including separate unintelligible varieties ) Morocco , Algeria Chewa Niger -- Congo
9,700,000 Malawi , Zimbabwe English Indo - European 6,500,
</item 9>

<item 10>
Languages of Africa
English , French and Portuguese are important languages in Africa . About 130 million , 115 million and 30 million Africans , respectively , speak them as
either native or secondary languages . Portuguese has become the national language of Angola . Through ( among other factors ) sheer demographic weight ,
Africans are increasingly taking ownership of these three world languages and having an ever greater influence on their development and growth . Linguistic
features ( edit ) Some linguistic features are particularly common among languages spoken in Africa , whereas others are less common . Such shared traits
probably are not due to a common origin of all African languages . Instead , some may be due to language contact ( resulting in borrowing ) and specific idioms
and phrases may be due to a similar cultural background . Phonological ( edit ) Some widespread phonetic features include : certain types of consonants , such
as implosives ( / ɓa / ) , ejectives ( / kʼa / ) , the labiodental flap and in southern Africa , clicks ( / ǂa / , / ŋǃa / ) . True implosives are rare outside
Africa , and clicks and the flap almost unheard of .
</item 10>
=== END ITEM id=54f6ec6f90331457 ===
