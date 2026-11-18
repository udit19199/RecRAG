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

=== ITEM id=c11f97f18e197a00 ===
Gold sentences and facts:
Sentence 0: Cast Character Rank / Position Seasons Notes Regular Recurring Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 , 17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 -- 5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D. Wong George Huang FBI Special Agent 4 -- 12 2 -- 3 13 -- 15 , 17 Diane Neal Casey Novak Senior Assistant District Attorney 5 -- 9 13 12 Adam Beach Chester Lake Detective 9 8 Michaela McManus Kim Greylek Assistant District Attorney 10 Danny Pino Nick Amaro Detective 13 -- 16 Kelli Giddish Amanda Rollins Detective 13 -- Raúl Esparza Rafael Barba Assistant District Attorney 15 -- 14 Peter Scanavino Dominick Carisi Jr . Detective 16 -- 16
  Fact 0.0: Christopher Meloni plays Elliot Stabler.
  Fact 0.1: Elliot Stabler is a Detective.
  Fact 0.2: Christopher Meloni's season entry is 1 to 12.
  Fact 0.3: Mariska Hargitay plays Olivia Benson.
  Fact 0.4: Olivia Benson is a Lieutenant.
  Fact 0.5: Mariska Hargitay's season entry starts at season 1.
  Fact 0.6: Richard Belzer plays John Munch.
  Fact 0.7: John Munch is a DA Investigator.
  Fact 0.8: Richard Belzer's season entries are 1 to 15 and 15 and 17.
  Fact 0.9: Dann Florek plays Donald Cragen.
  Fact 0.10: Donald Cragen is a Captain.
  Fact 0.11: Dann Florek's season entries are 1 to 15 and 16.
  Fact 0.12: Michelle Hurd plays Monique Jeffries.
  Fact 0.13: Monique Jeffries is a Detective.
  Fact 0.14: Michelle Hurd's season entry is 1 to 2.
  Fact 0.15: Stephanie March plays Alexandra Cabot.
  Fact 0.16: Alexandra Cabot is an Assistant District Attorney.
  Fact 0.17: Stephanie March's season entries are 2 to 5, 11, 2, 10, 13, and 6.
  Fact 0.18: Ice - T plays Odafin "Fin" Tutuola.
  Fact 0.19: Odafin "Fin" Tutuola is a Detective.
  Fact 0.20: Ice - T's season entry starts at season 2.
  Fact 0.21: Tamara Tunie plays Melinda Warner.
  Fact 0.22: Melinda Warner is a Medical Examiner.
  Fact 0.23: Tamara Tunie's season entries are 7 to 12, 2 to 6, 13 to 17, and 19.
  Fact 0.24: B.D. Wong plays George Huang.
  Fact 0.25: George Huang is an FBI Special Agent.
  Fact 0.26: B.D. Wong's season entries are 4 to 12, 2 to 3, 13 to 15, and 17.
  Fact 0.27: Diane Neal plays Casey Novak.
  Fact 0.28: Casey Novak is a Senior Assistant District Attorney.
  Fact 0.29: Diane Neal's season entries are 5 to 9, 13, and 12.
  Fact 0.30: Adam Beach plays Chester Lake.
  Fact 0.31: Chester Lake is a Detective.
  Fact 0.32: Adam Beach's season entries are 9 and 8.
  Fact 0.33: Michaela McManus plays Kim Greylek.
  Fact 0.34: Kim Greylek is an Assistant District Attorney.
  Fact 0.35: Michaela McManus's season entry is 10.
  Fact 0.36: Danny Pino plays Nick Amaro.
  Fact 0.37: Nick Amaro is a Detective.
  Fact 0.38: Danny Pino's season entry is 13 to 16.
  Fact 0.39: Kelli Giddish plays Amanda Rollins.
  Fact 0.40: Amanda Rollins is a Detective.
  Fact 0.41: Kelli Giddish's season entry starts at season 13.
  Fact 0.42: Raúl Esparza plays Rafael Barba.
  Fact 0.43: Rafael Barba is an Assistant District Attorney.
  Fact 0.44: Raúl Esparza's season entry is 15 to 14.
  Fact 0.45: Peter Scanavino plays Dominick Carisi Jr.
  Fact 0.46: Dominick Carisi Jr. is a Detective.
  Fact 0.47: Peter Scanavino's season entry is 16 to 16.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
The series follows a unit of investigators in Moscow whose job is to investigate crimes of a sexual nature . The series airs on NTV and is produced by Pavel
Korchagin , Felix Kleiman , and Edward Verzbovski and directed by Dmitry Brusnikin . The screenplays are written by Sergei Kuznvetsov , Elena Karavaeshnikova ,
and Maya Shapovalova . Casting ( edit ) Cast of Law & Order : Special Victims Unit Cast of seasons 4 -- 5 ( 2002 -- 04 ) , from left : Ice - T , Richard Belzer
, Mariska Hargitay , Dann Florek , Christopher Meloni , B.D.
</item 1>

<item 2>
Graph query row:
series: Law & Order: Special Victims Unit: The First Year
cast: [{'actor': 'Richard Belzer', 'character': 'Law & Order: Special Victims Unit', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor':
'Christopher Meloni', 'character': 'Detective Elliot Stabler', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Christopher Meloni',
'character': 'Elliot Stabler', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Christopher Meloni', 'character': 'Outstanding Lead Actor
in a Drama Series', 'role': None, 'fact': 'nominated in 2006', 'relationship': 'RELATED_TO'}, {'actor': 'Christopher Meloni', 'character': 'Law & Order: Special
Victims Unit', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Christopher Meloni', 'character': 'Warren Leight', 'role': None, 'fact':
None, 'relationship': 'RELATED_TO'}, {'actor': 'Christopher Meloni', 'character': 'Wolf', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor':
'Dann Florek', 'character': 'Exiled: A Law & Order Movie', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Dann Florek', 'character':
'Wolf', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Dann Florek', 'character': 'Law & Order: Special Victims Unit', 'role': None,
'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Dann Florek', 'character': 'Captain Don Cragen', 'role': None, 'fact': None, 'relationship':
'RELATED_TO'}, {'actor': 'Dann Florek', 'character': 'Law & Order', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Michelle Hurd',
'character': 'Monique Jeffries', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Michelle Hurd', 'character': 'Leap Years', 'role': None,
'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Michelle Hurd', 'character': 'Law & Order: Special Victims Unit', 'role': None, 'fact': None,
'relationship': 'RELATED_TO'}, {'actor': 'Ice-T', 'character': 'Exiled', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Ice-T',
'character': 'Wolf', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor': 'Ice-T', 'character': 'New York Undercover', 'role': None, 'fact':
None, 'relationship': 'RELATED_TO'}, {'actor': 'Ice-T', 'character': 'Odafin Tutuola', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, {'actor':
'Ice-T', 'character': 'Law & Order: Special Victims Unit (season 19)', 'role': None, 'fact': None, 'relationship': 'RELATED_TO'}, '(+62 more)']
</item 2>

<item 3>
Law & Order: Special Victims Unit
52 8.31 18 21 September 21 , 2016 ( 2016 - 09 - 21 ) 7.83 May 24 , 2017 ( 2017 - 05 - 24 ) 6.22 2016 -- 17 No. 48 7.39 19 24 September 27 , 2017 ( 2017 - 09 -
27 ) 5.67 TBA TBA 2017 - 18 TBA TBA Awards and honors ( edit ) Main article : List of awards and nominations received by Law & Order : Special Victims Unit Law
& Order : Special Victims Unit has received many awards and award nominations . Mariska Hargitay has twice been nominated for a Golden Globe Award and won once
in 2005 . The show has been nominated numerous times for the Emmy Award . Mariska Hargitay has been nominated for the Outstanding Lead Actress in a Drama Series
category 8 years in a row beginning in 2004 and won the Emmy in 2006 . Christopher Meloni was nominated for the Outstanding Lead Actor in a Drama Series
category in 2006 .
</item 3>

<item 4>
Law & Order: Special Victims Unit
The unit also works with the Manhattan District Attorney 's office as they prosecute cases and seek justice for SVU 's victims and survivors with precision and
a passion to win and bring closure to the intense investigations . The series often uses stories that are `` ripped from the headlines '' or based on real
crimes . Such episodes take a real crime and fictionalize it by changing some details . Originally the show focused around the detective pairings of Elliot
Stabler & Olivia Benson and John Munch & Brian Cassidy . Stabler is a seasoned veteran of the unit who has seen it all and tries his best to protect his family
from the horrors he sees every day . His partner Benson 's difficult past as the child of a rape victim is the reason she joined the unit . Backing them up is
John Munch , and his first partner Brian Cassidy . Munch is a transfer from Baltimore 's homicide unit , who brings his acerbic wit , conspiracy theories , and
street - honed investigative skills . Whilst Cassidy is young and eager to learn from his fellow detectives . These two detective teams received support from
Detectives Monique Jeffries and Ken Briscoe .
</item 4>

<item 5>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 5>

<item 6>
Law & Order: Special Victims Unit
Series overview ( edit ) In the criminal justice system , sexually based offenses are considered especially heinous . In New York City , the dedicated
detectives who investigate these vicious felonies are members of an elite squad known as the Special Victims Unit . These are their stories . -- Opening
narration spoken by Steven Zirnkilton Based out of the New York City Police Department 's 16th precinct in Manhattan , Law & Order : Special Victims Unit delves
into the dark side of the New York underworld as the detectives of a new elite force , the Special Victims Unit ( SVU for short ) , investigate and prosecute
various sexually - oriented crimes including rape , pedophilia , and domestic violence . They also investigate the abuses of children , the disabled and elderly
victims of non-sexual crimes who require specialist handling . All the while trying to balance the effects of the investigation on their own lives . Its stories
also touch on the political and societal issues associated with gender identity , sexual preferences , and equality rights . While the victim is often murdered
, this is not always the case , and victims frequently play prominent roles in episodes .
</item 6>

<item 7>
Law & Order: Special Victims Unit
Cast and characters ( edit ) Main article : List of Law & Order : Special Victims Unit characters Cast Character Rank / Position Seasons Notes Regular Recurring
Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 ,
17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 --
5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D.
</item 7>

<item 8>
Law & Order: Special Victims Unit
It is the current longest running scripted non-animated U.S. primetime TV series since the cancellation of the original Law & Order on May 24 , 2010 and is the
fourth - longest running scripted U.S. primetime TV series on a major broadcast network . Law & Order : Special Victims Unit originally centered almost
exclusively on the detectives of the Special Victims Unit in a fictional version of the 16th Precinct of the New York City Police Department . As the series
progressed , additional supporting characters were added as allies of the detectives in the New York County Manhattan District Attorney 's office ( known as
advisers from the Sex Crimes Bureau ) and the Medical Examiner 's office . Certain episodes will go into detail about detectives ' personal lives and how they
may or may not tie into the crimes dealt with during the show . Typical episodes follow the detectives and their colleagues as they investigate and prosecute
sexually based offenses , child abuse , and crimes against the elderly . The show starred Christopher Meloni as Detective Elliot Stabler and Mariska Hargitay as
Detective Olivia Benson for its first twelve seasons until the former left the cast , unable to come to an agreement on his contract .
</item 8>

<item 9>
Law & Order: Special Victims Unit
External links ( edit ) Wikimedia Commons has media related to Law & Order : Special Victims Unit . Wikiquote has quotations related to : Law & Order : Special
Victims Unit Official website on NBC Law & Order : Special Victims Unit on IMDb Law & Order : Special Victims Unit at TV.com `` List of Law & Order : SVU
Episodes '' . TV Guide . Law & Order : Special Victims Unit on MyNetworkTV Law & Order franchise Series Law & Order ( 1990 -- 2010 ) Law & Order : Special
Victims Unit ( 1999 -- present ) Law & Order : Criminal Intent ( 2001 -- 2011 ) Law & Order : Trial by Jury ( 2005 -- 2006 ) Law & Order : LA ( 2010 -- 2011 )
Law & Order True Crime ( 2017 -- present ) Adaptations Paris enquêtes criminelles ( 2007 -- 2008 ) Law & Order : UK ( 2009 -- 2014 ) Video games Law & Order :
Dead on the Money Law & Order : Double or Nothing Law & Order : Justice Is Served Law & Order : Criminal Intent Law & Order : Legacies Related series Homicide :
</item 9>

<item 10>
Law & Order: Special Victims Unit
1999 ( 1999 - 09 - 20 ) -- present Chronology Related shows Law & Order franchise External links Website Law & Order : Special Victims Unit ( often abbreviated
to Law & Order : SVU or just SVU ) is an American police procedural , legal , crime drama television series set in New York City , where it is also primarily
produced . In the style of the original Law & Order , episodes are often `` ripped from the headlines '' or loosely based on real crimes that have received
media attention . Created by Dick Wolf , the series premiered on NBC on September 20 , 1999 , as the second series in Wolf 's successful Law & Order franchise .
Law & Order : Special Victims Unit has been nominated for and won numerous awards , including the 2006 Emmy for Outstanding Lead Actress in a Drama Series for
Mariska Hargitay , the first , and to date only , Emmy to be received by a regular on any Law & Order series . It is the current longest running scripted
non-animated U.S.
</item 10>
=== END ITEM id=c11f97f18e197a00 ===

=== ITEM id=aa54d841925592d8 ===
Gold sentences and facts:
Sentence 0: Hotel barge Bed and breakfast Botel Boutique hotel Bunkhouse
  Fact 0.0: Hotel barge is one of the listed terms.
  Fact 0.1: Bed and breakfast is one of the listed terms.
  Fact 0.2: Botel is one of the listed terms.
  Fact 0.3: Boutique hotel is one of the listed terms.
  Fact 0.4: Bunkhouse is one of the listed terms.

Retrieved items:
<item 1>
Graph query row:
entity: Bed and breakfast
relationship: PART_OF
connected_entity: Hotel types
</item 1>

<item 2>
Bed
Some inexpensive hostels provide bunk beds for guests . Bunk beds are used for children in private homes . A loft bed is similar to a bunk bed , except there is
no lower bunk . This leaves space underneath for storage , other furniture , a desk etc . A captain 's bed ( also known as a `` captain bed '' , `` chest bed ''
, or `` cabin bed '' ) is a platform bed with drawers and storage compartments built in underneath . A camp bed ( also `` cot '' ) is a simple , temporary ,
portable bed used by armies and by campers . Cots are also used to provide a sleeping surface for refugees and other homeless people during disasters , floods ,
or other crises . A canopy bed is similar to a four poster bed , but the posts usually extend higher and are adorned or draped with cloth , sometimes completely
enclosing the bed . A curtained bed is a luxury bed with curtains . A daybed is a couch that is used as a seat by day and as a bed by night . Usually it has pop
up trundle which is used as bed in the night . A futon is a traditional style of Japanese bed using a mattress on a wooden frame .
</item 2>

<item 3>
Disney's Art of Animation Resort
TheDailyDisney.com . Retrieved June 6 , 2010 . ^ Jump up to : `` More details on Disney 's Art of Animation Resort '' . TheDisneyBlog.com . Retrieved June 10 ,
2010 . Jump up ^ `` Disney 's Art of Animation Resort Unveiled `` Disney Parks Blog '' . Disneyparks.disney.go.com . May 12 , 2010 . Retrieved June 6 , 2010 .
Jump up ^ `` First look at new Art of Animation Resort concept art and rooms shown at Destination D: Walt Disney World '' . Inside the Magic . External links (
edit ) Official website Art of Animation Photo Gallery at HanBan Photos Walt Disney World Resorts Location Deluxe Moderate Value Deluxe Villa / Disney Vacation
Club Magic Kingdom Resort Area Disney 's Contemporary Resort Disney 's Grand Floridian Resort & Spa Disney 's Polynesian Village Resort Disney 's Wilderness
Lodge None None Bay Lake Tower at Disney 's Contemporary Resort Disney 's Polynesian Villas & Bungalows The Villas at Disney 's Grand Floridian Resort & Spa
Boulder Ridge Villas at Disney 's Wilderness Lodge Copper Creek Villas &
</item 3>

<item 4>
Disney's Art of Animation Resort
Floridian Resort & Spa Boulder Ridge Villas at Disney 's Wilderness Lodge Copper Creek Villas & Cabins at Disney 's Wilderness Lodge Epcot Resort Area Disney 's
Beach Club Resort Disney 's BoardWalk Inn Disney 's Yacht Club Resort Star Wars Hotel Disney 's Caribbean Beach Resort None Disney 's Beach Club Villas Disney
's BoardWalk Villas Disney Riviera Resort Animal Kingdom Resort Area Disney 's Animal Kingdom Lodge Disney 's Coronado Springs Resort Disney 's All - Star
Movies Resort Disney 's All - Star Music Resort Disney 's All - Star Sports Resort Disney 's Animal Kingdom Villas Disney Springs Resort Area None Disney 's
Port Orleans Resort - French Quarter Disney 's Port Orleans Resort - Riverside None Disney 's Old Key West Resort Disney 's Saratoga Springs Resort & Spa ESPN
Wide World of Sports Resort Area None None Disney 's Art of Animation Resort Disney 's Pop Century Resort None Campsites and cabins Disney 's Fort Wilderness
Resort & Campground Residental Golden Oak at Walt Disney World Resort Non-Disney Resorts B Resort & Spa Best Western Buena Vista Palace Doubletree Four Seasons
Hilton Holiday Inn Shades of Green Walt Disney World Dolphin Walt Disney World Swan Wyndham
</item 4>

<item 5>
Category:Hotel types
This list may not reflect recent changes ( learn more ) . Albergo Diffuso Apartment hotel Hotel barge Bed and breakfast Botel Boutique hotel Bunkhouse Capsule
hotel Caravanserai Casa particular Casino hotel Choultry Condo hotel Conference and resort hotels Eco hotel Extended stay hotel Flophouse Garden hotels Gasthaus
Guest house Heuhotel Holiday cottage Hostal Hostel Ice hotel Inn Love hotel Motel Patient hotel Pension ( lodging ) Pension hotel Pentahotels Pop - up hotel
Roadhouse ( facility ) Ryokan ( inn ) Serviced apartment Hotelship Stopping house Transit hotel Turbaza Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Category:Hotel_types&oldid=766350044 '' Categories : Hotels Talk Help About Wikipedia Azərbaycanca Deutsch Español
Euskara فارسی Français 한국어 Ido Magyar Polski Português Русский ไทย Türkçe اردو 中文 Edit links This page was last edited on 19 February 2017 , at 18 : 19 . About
Wikipedia
</item 5>

<item 6>
Dead Sea
Tourism and leisure Ein Bokek , a resort on the Israeli shore British mandate period A golf course named for Sodom and Gomorrah was built by the British at
Kalia on the northern shore . Israel The first major Israeli hotels were built in nearby Arad , and since the 1960s at the Ein Bokek resort complex . Israel has
15 hotels along the Dead Sea shore , generating total revenues of $291 million in 2012 . Most Israeli hotels and resorts on the Dead Sea are on a six -
kilometre ( 3.7 - mile ) stretch of the southern shore . Jordan Kempinski Hotel , one of the many hotels on the Jordanian shore On the Jordanian side , nine
international franchises have opened seaside resort hotels near the King Hussein Bin Talal Convention Center , along with resort apartments , on the eastern
shore of the Dead Sea . The 9 hotels have boosted the Jordanian side 's capacity to 2,800 rooms . On November 22 , 2015 , the Dead Sea panorama road was
included along with 40 archaeological locations in Jordan , to become live on Google Street View .
</item 6>

<item 7>
Category:Hotel types
Category : Hotel types - wikipedia Help Category : Hotel types Jump to : navigation , search Wikimedia Commons has media related to Hotels by type . The main
article for this category is Hotel . Pages in category `` Hotel types '' The following 40 pages are in this category , out of 40 total . This list may not
reflect recent changes ( learn more ) .
</item 7>

<item 8>
Graph query row:
entity: Boutique hotel
relationship: PART_OF
connected_entity: Hotel types
</item 8>

<item 9>
Graph query row:
entity: Motel
relationship: PART_OF
connected_entity: Hotel types
</item 9>

<item 10>
Disney's Art of Animation Resort
Theming ( edit ) The resort is designed `` with families in mind . '' It features four of Disney 's popular character themes : Cars , Finding Nemo , The Lion
King , and The Little Mermaid . Much like the other Disney Value Resorts , giant versions of various items are built around the hotel on each of the ten wings ,
such as a 35 - foot ( 11 m ) model of King Triton . In total , there are 1,984 rooms. of which 1,120 are family suites capable of housing up to six people ,
featuring living rooms and bedrooms . The remaining 864 sport the standard value layout . The resort has a total of ten buildings , as well as three themed
pools . Finding Nemo : The Finding Nemo section of the resort is themed with the ocean setting featured in the film , with underwater plant and animal
decorations throughout the buildings . This is the first area of the resort , which opened on May 31 , 2012 . Cars : The Cars section of the resort is themed
like the Cozy Cone Motel that has featured in the film .
</item 10>
=== END ITEM id=aa54d841925592d8 ===

=== ITEM id=bf36b6bb633c83de ===
Gold sentences and facts:
Sentence 0: An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine .
  Fact 0.0: An elevator with a counterbalance approximates an ideal Atwood machine.
  Fact 0.1: A counterbalanced elevator relieves the driving motor from the load of holding the elevator cab.
  Fact 0.2: A counterbalanced elevator's driving motor has to overcome only the weight difference and inertia of the two masses.
  Fact 0.3: The same principle is used for funicular railways with two connected railway cars on inclined tracks.
  Fact 0.4: The same principle is used for the elevators on the Eiffel Tower, which counterbalance each other.
  Fact 0.5: Ski lifts are another example.
  Fact 0.6: In ski lifts, the gondolas move on a closed (continuous) pulley system up and down the mountain.
  Fact 0.7: A ski lift is similar to the counter-weighted elevator, but with a constraining force provided by the cable in the vertical dimension.
  Fact 0.8: A ski lift achieves work in both the horizontal and vertical dimensions.
  Fact 0.9: Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine.

Retrieved items:
<item 1>
Atwood machine
The angular acceleration of the pulley is given by the no - slip condition : α = a r , ( \ displaystyle \ alpha = ( a \ over r ) , ) where α ( \ displaystyle \
alpha ) is the angular acceleration . The net torque is then : τ n e t = ( T 1 − T 2 ) r − τ f r i c t i o n = I α ( \ displaystyle \ tau _ ( \ mathrm ( net ) )
= \ left ( T_ ( 1 ) - T_ ( 2 ) \ right ) r - \ tau _ ( \ mathrm ( friction ) ) = I \ alpha ) Combining with Newton 's second law for the hanging masses , and
solving for T , T , and a , we get : Acceleration : a = g ( m 1 − m 2 ) − τ f r i c t i o n r m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 )
) - ( \ tau _ ( \ mathrm ( friction ) ) \ over r ) ) \ over ( m_ (
</item 1>

<item 2>
Atwood machine
To evaluate tension , substitute the equation for acceleration in either of the 2 force equations . a = g m 1 − m 2 m 1 + m 2 ( \ displaystyle a = g ( m_ ( 1 )
- m_ ( 2 ) \ over m_ ( 1 ) + m_ ( 2 ) ) ) For example , substituting into m 1 a = m 1 g − T ( \ displaystyle m_ ( 1 ) a = m_ ( 1 ) g-T ) , results in T = 2 g m
1 m 2 m 1 + m 2 = 2 g 1 / m 1 + 1 / m 2 ( \ displaystyle T = ( 2gm_ ( 1 ) m_ ( 2 ) \ over m_ ( 1 ) + m_ ( 2 ) ) = ( 2g \ over 1 / m_ ( 1 ) + 1 / m_ ( 2 ) ) )
Equations for a pulley with inertia and friction ( edit ) For very small mass differences between m and m , the rotational inertia I of the pulley of radius r
can not be neglected .
</item 2>

<item 3>
Atwood machine
) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment nearest m : T 1 = m 1 g ( 2 m 2 + I r 2 ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 1 ) = ( (
m_ ( 1 ) g ( 2m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Tension in string segment
nearest m : T 2 = m 2 g ( 2 m 1 + I r 2 ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) \ over (
m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \
</item 3>

<item 4>
Atwood machine
ISBN 81 - 85015 - 53 - 8 . Section 1 - 6 , example 2 Jump up ^ See Wikipedia article : https://en.wikipedia.org/wiki/Boat_lift External links ( edit ) Professor
Greenslade 's account on the Atwood Machine Atwood 's Machine by Enrique Zeleny , The Wolfram Demonstrations Project . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Atwood_machine&oldid=792136529 '' Categories : Mechanics Physics experiments Talk Contents About Wikipedia Català
Deutsch Español Français 한국어 Hrvatski Italiano Nederlands 日本 語 Polski Português Русский Srpskohrvatski / српскохрватски Türkçe Українська 中文 Edit links This
page was last edited on 24 July 2017 , at 17 : 58 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy .
</item 4>

<item 5>
Atwood machine
Boat lifts are another type of counter-weighted elevator system approximating an Atwood machine . See also ( edit ) Wikimedia Commons has media related to
Atwood 's machine . Frictionless plane Kater 's pendulum Spherical cow Swinging Atwood 's machine Notes ( edit ) Jump up ^ Tipler , Paul A. ( 1991 ) . Physics
For Scientists and Engineers ( 3rd , extended ed . ) . New York : Worth Publishers . p. 160 . ISBN 0 - 87901 - 432 - 6 . Chapter 6 , example 6 - 13 Jump up ^
Goldstein , Herbert ( 1980 ) . Classical Mechanics ( 2nd ed . ) . New Delhi : Addison - Wesley / Narosa Indian Student Edition . pp. 26 -- 27 . ISBN 81 - 85015
- 53 - 8 .
</item 5>

<item 6>
Atwood machine
+ τ f r i c t i o n r g ) m 1 + m 2 + I r 2 ( \ displaystyle T_ ( 2 ) = ( ( m_ ( 2 ) g ( 2m_ ( 1 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) + ( ( \ tau _ ( \ mathrm (
friction ) ) ) \ over ( rg ) ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Should bearing friction be negligible ( but not the
inertia of the pulley and not the traction of the string on the pulley rim ) , these equations simplify as the following results : Acceleration : a = g ( m 1 −
m 2 ) m 1 + m 2 + I r 2 ( \ displaystyle a = ( ( g ( m_ ( 1 ) - m_ ( 2 ) ) ) \ over ( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) )
</item 6>

<item 7>
Atwood machine
( m_ ( 1 ) + m_ ( 2 ) + ( ( I ) \ over ( r ^ ( 2 ) ) ) ) ) ) Practical implementations ( edit ) Atwood 's original illustrations show the main pulley 's axle
resting on the rims of another four wheels , to minimize friction forces from the bearings . Many historical implementations of the machine follow this design .
An elevator with a counterbalance approximates an ideal Atwood machine and thereby relieves the driving motor from the load of holding the elevator cab -- it
has to overcome only weight difference and inertia of the two masses . The same principle is used for funicular railways with two connected railway cars on
inclined tracks , and for the elevators on the Eiffel Tower which counterbalance each other . Ski lifts are another example , where the gondolas move on a
closed ( continuous ) pulley system up and down the mountain . The ski lift is similar to the counter-weighted elevator , but with a constraining force provided
by the cable in the vertical dimension thereby achieving work in both the horizontal and vertical dimensions . Boat lifts are another type of counter-weighted
elevator system approximating an Atwood machine .
</item 7>

<item 8>
Atwood machine
Atwood machine - wikipedia Atwood machine Jump to : navigation , search Illustration of Atwood machine , 1905 . The Atwood machine ( or Atwood 's machine ) was
invented in 1784 by the English mathematician George Atwood as a laboratory experiment to verify the mechanical laws of motion with constant acceleration .
Atwood 's machine is a common classroom demonstration used to illustrate principles of classical mechanics . The ideal Atwood Machine consists of two objects of
mass m and m , connected by an inextensible massless string over an ideal massless pulley . When m = m , the machine is in neutral equilibrium regardless of the
position of the weights . When m ≠ m both masses experience uniform acceleration . Contents ( hide ) 1 Equation for constant acceleration 2 Equation for tension
3 Equations for a pulley with inertia and friction 3.1 Practical implementations 4 See also 5 Notes 6 External links Equation for constant acceleration ( edit )
The free body diagrams of the two hanging masses of the Atwood machine .
</item 8>

<item 9>
Atwood machine
Weight of m 1 ( \ displaystyle m_ ( 1 ) ) and m 2 ( \ displaystyle m_ ( 2 ) ) is simply W 1 = m 1 g ( \ displaystyle W_ ( 1 ) = m_ ( 1 ) g ) and W 2 = m 2 g ( \
displaystyle W_ ( 2 ) = m_ ( 2 ) g ) respectively . Forces affecting m : m 1 g − T = m 1 a ( \ displaystyle \ ; m_ ( 1 ) g-T = m_ ( 1 ) a ) Forces affecting m :
T − m 2 g = m 2 a ( \ displaystyle \ ; T - m_ ( 2 ) g = m_ ( 2 ) a ) and adding the two previous equations we obtain m 1 g − m 2 g = m 1 a + m 2 a ( \
displaystyle \ ; m_ ( 1 ) g - m_ ( 2 ) g = m_ ( 1 ) a + m_ ( 2 ) a ) ,
</item 9>

<item 10>
Atwood machine
and our concluding formula for acceleration a = g m 1 − m 2 m 1 + m 2 ( \ displaystyle a = g ( m_ ( 1 ) - m_ ( 2 ) \ over m_ ( 1 ) + m_ ( 2 ) ) ) Conversely ,
the acceleration due to gravity , g , can be found by timing the movement of the weights , and calculating a value for the uniform acceleration a : a = 2 d t 2
( \ displaystyle a = ( \ frac ( 2d ) ( t ^ ( 2 ) ) ) ) . The Atwood machine is sometimes used to illustrate the Lagrangian method of deriving equations of
motion . Equation for tension ( edit ) It can be useful to know an equation for the tension in the string . To evaluate tension , substitute the equation for
acceleration in either of the 2 force equations .
</item 10>
=== END ITEM id=bf36b6bb633c83de ===

=== ITEM id=c8c3835159b9f192 ===
Gold sentences and facts:
Sentence 0: A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie , starring Jane Fonda and Lily Tomlin .
  Fact 0.0: A cover of a song by Stacey Markus is used at the end of episode nine, 'The Invitation', of Grace and Frankie.
  Fact 0.1: Grace and Frankie is a 2015 Netflix original series.
  Fact 0.2: Grace and Frankie stars Jane Fonda.
  Fact 0.3: Grace and Frankie stars Lily Tomlin.

Retrieved items:
<item 1>
Let My Love Open the Door
It reached number five in Canada . Contents ( hide ) 1 Background 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 Personnel 4 Covers 5 List of
Films , TV Shows , and Other Media featuring the track 6 References 7 External links Background ( edit ) Although Townshend is a devotee to the religious guru
Meher Baba , he claimed in the liner notes of his Anthology CD that `` Jesus sings '' on the track . `` Let My Love Open the Door '' was released as the second
single from Empty Glass in Britain , where it was backed with the non-album tracks `` Classified '' and `` Greyhound Girl . '' The song was a minor British hit
, reaching # 46 . The song saw more success when it was released as the debut single from Empty Glass in America , where the song reached # 9 . It was Pete
Townshend 's only solo top 10 hit on the Billboard Hot 100 , but The Who 's song `` I Can See For Miles , '' which was written by Townshend , reached the same
position on the chart 13 years earlier .
</item 1>

<item 2>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 2>

<item 3>
Let My Love Open the Door
Deeds ( 2002 ) Jersey Girl as `` E. Cola Mix '' ( 2004 ) Along Came Polly ( 2004 ) Click ( in trailers ) ( 2006 ) Dan in Real Life ( 2007 ) Doctors ( soap opera
) , season 11 , episode 106 ( 2009 ) Old Dogs ( 2009 ) How Do You Know ( in trailers ) ( 2010 ) Take Me Home Tonight as `` E. Cola Mix '' ( 2011 ) Red Dog (
2011 ) Hit and Run ( 2012 ) The Newsroom ( 2012 ) - cover by Luminate used in the second season finale , `` Election Night , Part II '' ( September 15 , 2013 )
Californication , season 7 , episode 1 ( April 13 , 2014 ) The Goldbergs , season 1 `` Why 're You Hitting Yourself ? '' ( 2013 ) the `` E. Cola Mix '' played
at end before credits . Also played in Season 4 , Episode 15 `` So Swayze It 's Crazy '' ( 2017 ) .
</item 3>

<item 4>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 4>

<item 5>
Let My Love Open the Door
The group Rogue Wave also covered it in `` Cover me '' released in 2017 . List of films , TV shows , and other media featuring the track ( edit ) The song has
been used frequently in film , most notably the comedy genre , often as trailer music for movies such as Jerry Maguire and How Do You Know , and then as end
credit song for Mr. Deeds in 2002 ; Jersey Girl and Along Came Polly , both in 2004 . It was featured in the following films and TV series : Look Who 's Talking
( 1989 ) Grosse Pointe Blank as `` E. Cola Mix '' ( 1997 ) Mr.
</item 5>

<item 6>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 6>

<item 7>
Let My Love Open the Door
Psychoderelict O Parvardigar Related articles Songs Songs Written Albums Peter Meaden The Who Deep End Tommy ( film ) The Who 's Tommy Tommy ( EP ) Tommy (
soundtrack ) The Boy Who Heard Music Quadrophenia ( musical ) Quadrophenia ( film ) Quadrophenia ( soundtrack ) The Lifehouse Method Horse 's Neck Who I Am Eel
Pie Publishing Eel Pie Studios Double O Cliff Townshend Emma Townshend Simon Townshend The Wick Ashdown House , Oxfordshire The Boathouse , Twickenham Chapel
House , Twickenham Retrieved from `` https://en.wikipedia.org/w/index.php?title=Let_My_Love_Open_the_Door&oldid=838422442 '' Categories : 1980 singles Debut
singles Pete Townshend songs Songs written by Pete Townshend 1980 songs Atco Records singles Song recordings produced by Pete Townshend British new wave songs
Hidden categories : Webarchive template wayback links Music infoboxes with deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia
Čeština Español Euskara Edit links This page was last edited on 26 April 2018 ,
</item 7>

<item 8>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 8>

<item 9>
Let My Love Open the Door
A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie ,
starring Jane Fonda and Lily Tomlin . The song is featured in a Walmart Christmas holiday television commercial which started airing in early December 2017 .
Lethal Weapon ( TV series ) , season 2 , episode 9 `` Fools Rush In '' play the song performed by `` Rogue wave '' ( 2017 ) . References ( edit ) Jump up ^
Unterberger , Richie ( 2012 - 06 - 14 ) . `` Let My Love Open the Door '' . Rolling Stone . Retrieved 2016 - 10 - 16 . Jump up ^ `` Pete Townshend , ' Empty
Glass ' 100 Best Albums of the Eighties '' . Rolling Stone . 1989 - 11 - 16 . Retrieved 2016 - 10 - 16 . ^ Jump up to : Bac-lac.gc.ca ^ Jump up to : `` Let My
Love Open the Door '' . Songfacts.com .
</item 9>

<item 10>
When Calls the Heart
Rosemary even came to like Elizabeth , even if only on an acquaintance level . Truthfully , she cares about everyone in Hope Valley , especially Lee , with whom
she shares an mutual attraction . When Rosie is not spending time with Lee , talking to Abigail , or dreaming about the glamorous life of an actress , she can
often be found somewhere writing her weekly advice column for Hope Valley 's newspaper ! In season 3 , Rosemary 's friendship with Elizabeth deepened . Instead
of viewing her as an adversary , she finds a friend in Elizabeth . Throughout season 3 , Rosemary 's relationship with Lee becomes more romantic . In `` Heart
Of A Hero , '' seeing that Lee needs to spend more time at the sawmill to help the new settlers find jobs , Rosemary tells Lee that the theatre does n't have to
be built . He insists , but she says how important his job is . He smiles and tells her he will one day build her theatre , telling her that she is `` the best
part of his life . '' The couple shares in their first kiss . In `` A Gentle Heart , '' Lee is injured .
</item 10>
=== END ITEM id=c8c3835159b9f192 ===

=== ITEM id=16e1f06cce875bd7 ===
Gold sentences and facts:
Sentence 0: The Regulatory Reform ( Fire Safety ) Order 2005 ( officially listed as The Regulatory Reform ( Fire Safety ) Order 2005 S.I. 2005 No. 1541 ) is a statutory instrument , applicable only in England and Wales . The Order places the responsibility on individuals within an organisation to carry out risk assessments to identify , manage and reduce the risk of fire . The Order was made into law on 7 June 2005 and came into force on 1 October 2006 .
  Fact 0.0: The Regulatory Reform (Fire Safety) Order 2005 is officially listed as The Regulatory Reform (Fire Safety) Order 2005 S.I. 2005 No. 1541.
  Fact 0.1: The Regulatory Reform (Fire Safety) Order 2005 is a statutory instrument.
  Fact 0.2: The Regulatory Reform (Fire Safety) Order 2005 applies only in England and Wales.
  Fact 0.3: The Regulatory Reform (Fire Safety) Order 2005 places the responsibility on individuals within an organisation to carry out risk assessments.
  Fact 0.4: The risk assessments required by the Regulatory Reform (Fire Safety) Order 2005 identify, manage and reduce the risk of fire.
  Fact 0.5: The Regulatory Reform (Fire Safety) Order 2005 was made into law on 7 June 2005.
  Fact 0.6: The Regulatory Reform (Fire Safety) Order 2005 came into force on 1 October 2006.

Retrieved items:
<item 1>
Indian High Courts Act 1861
1994 The Prohibition of Child Marriage Act , 2006 The Scheduled Tribes and Other Traditional Forest Dwellers ( Recognition of Forest Rights ) Act , 2006 Street
Vendors Act , 2014 State Laws Anti-Superstition and Black Magic Act The Bombay Prohibition Act , 1949 Bombay Prohibition ( Gujarat Amendment ) 2009 Chhattisgarh
Food Security Act , 2012 Maharashtra Control of Organised Crime Act The Gujarat Local Authorities Laws ( Amendment ) Bill , 2009 Puducherry Prevention of
Anti-Social Activities Act State Reorganisation Andhra Pradesh Reorganisation Act , 2014 Bihar Reorganisation Act , 2000 Madhya Pradesh Reorganisation Act ,
2000 Punjab Reorganisation Act , 1966 States Reorganisation Act , 1956 Uttar Pradesh Reorganisation Act , 2000 Terrorism Unlawful Activities ( Prevention ) Act
Transport Aircraft Act Motor Vehicles Act , 1988 National Waterways Act , 2016 Inland Vessels Act Metro Railways Act , 1978 Metro Railway Act , 2002
Organisation / Body Chartered Accountants Act , 1949 Companies Act 2013 Gram Nyayalayas Act ,
</item 1>

<item 2>
Central London Property Trust Ltd v High Trees House Ltd
In my opinion , the time has now come for the validity of such a promise to be recognized . The logical consequence , no doubt is that a promise to accept a
smaller sum in discharge of a larger sum , if acted upon , is binding notwithstanding the absence of consideration : and if the fusion of law and equity leads
to this result , so much the better . That aspect was not considered in Foakes v. Beer ( 1884 ) 9 App . Cas. 605 . At this time of day however , when law and
equity have been joined together for over seventy years , principles must be reconsidered in the light of their combined effect . It is to be noticed that in
the Sixth Interim Report of the Law Revision Committee , pars. 35 , 40 , it is recommended that such a promise as that to which I have referred should be
enforceable in law even though no consideration for it has been given by the promisee . It seems to me that , to the extent I have mentioned , that result has
now been achieved by the decisions of the courts .
</item 2>

<item 3>
Regulatory Reform (Fire Safety) Order 2005
wikipedia.org/w/index.php?title=Regulatory_Reform_(Fire_Safety)_Order_2005&oldid=636696481 '' Categories : Fire and rescue in the United Kingdom Statutory
Instruments of the United Kingdom 2005 in British law Talk Contents About Wikipedia Add links This page was last edited on 5 December 2014 , at 01 : 52 . Text
is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Regulatory Reform (Fire Safety) Order 2005
Regulatory Reform ( Fire Safety ) Order 2005 - wikipedia Regulatory Reform ( Fire Safety ) Order 2005 Jump to : navigation , search The Regulatory Reform ( Fire
Safety ) Order 2005 ( officially listed as The Regulatory Reform ( Fire Safety ) Order 2005 S.I. 2005 No. 1541 ) is a statutory instrument , applicable only in
England and Wales . The Order places the responsibility on individuals within an organisation to carry out risk assessments to identify , manage and reduce the
risk of fire . The Order was made into law on 7 June 2005 and came into force on 1 October 2006 . Guidance for Businesses and Organisation is available in the
form of 16 Government Published documents , with general guidance , a 5 - Step Checklist and 12 documents pertaining specifically to a particular type of
business premises . It does not normally apply to domestic premises . Contents ( hide ) 1 Guides 2 See also 3 References 4 External links Guides ( edit ) The
guides available are Do you have paying guests ?
</item 4>

<item 5>
Regulatory Reform (Fire Safety) Order 2005
Fire safety risk assessment : animal premises and stables Fire safety risk assessment : means of escape for disabled people Fire safety risk assessment : open -
air events and venues Fire safety risk assessment : transport premises and facilities Fire safety risk assessment : healthcare premises Fire safety risk
assessment : residential care premises Fire safety risk assessment : theatres , cinemas and similar premises Making your premises safe from fire Fire safety
risk assessment : 5 - step checklist Fire safety risk assessment : factories and warehouses Fire safety risk assessment : large places of assembly Fire safety
risk assessment : small and medium places of assembly Fire safety risk assessment : educational premises Fire safety risk assessment : sleeping accommodation
Fire safety risk assessment : offices and shops See also ( edit ) Fire services in the United Kingdom References ( edit ) Jump up ^ The Regulatory Reform ( Fire
Safety ) Order 2005 , England & Wales Retrieved on 20 March 2014 Jump up ^ The Regulatory Reform ( Fire Safety ) Order 2005 , England & Wales Retrieved on 20
march 2014 Jump up ^ Fire safety law and guidance documents for business Retrieved on 20 march 2014 External links ( edit ) The Regulatory Reform ( Fire Safety
) Order 2005 Retrieved from `` https://en.
</item 5>

<item 6>
United Kingdom corporation tax
, HMRC , 2003 . Retrieved 17 April 2007 Jump up ^ Corporation tax reform -- a technical note ( PDF ) Archived 21 February 2007 at the Wayback Machine. , HMRC ,
2004 . Retrieved 17 April 2007 Jump up ^ Finance Act 2002 , HMSO , ISBN 0 - 10 - 542302 - 5 `` Company Taxation Manual '' . HMRC . Retrieved 17 May 2007 . ``
Income and Corporation Taxes Act 1988 '' . HMSO. 1988 . Retrieved 17 May 2007 . `` Taxation of Chargeable Gains Act 1992 '' . HMSO. 1992 . Retrieved 17 May 2007
. `` Capital Allowances Act 2001 '' . HMSO. 2001 . Retrieved 17 May 2007 . Finance Acts ( 1965 -- ) , HMSO .
</item 6>

<item 7>
Indian High Courts Act 1861
2015 National Security Act ( India ) Protection of Women from Domestic Violence Act , 2005 Scheduled Caste and Scheduled Tribe ( Prevention of Atrocities ) Act
, 1989 Education Indian Institutes of Management Act , 2017 Institutes of Technology Act , 1961 National Institutes of Technology Act , 2007 Right of Children
to Free and Compulsory Education Act , 2009 University Grants Commission Act , 1956 Environment Air ( Prevention and Control of Pollution ) Act Biological
Diversity Act , 2002 CAMPA bill Environment Protection Act , 1986 Indian Forest Act , 1927 National Green Tribunal Act Prevention of Cruelty to Animals Act
Protection of Plant Varieties and Farmers ' Rights Act , 2001 Wildlife Protection Act , 1972 Financial Aadhaar ( Targeted Delivery of Financial and other
Subsidies , benefits and services ) Act , 2016 Banking Regulation Act , 1949 Conservation of Foreign Exchange and Prevention of Smuggling Activities Act
Depositories Act The Electricity Act , 2003 Expenditure Tax Act , 1987 Finance Act ( India ) Fiscal Responsibility and Budget Management Act ,
</item 7>

<item 8>
United Kingdom corporation tax
Finance Acts ( 1965 -- ) , HMSO . From 1998 onwards , consolidation took place , into the Income and Corporation Taxes Act 1988 , the Taxation of Chargeable
Gains Act and the Capital Allowances Act 2001 . `` Reform of corporation tax -- A consultation document '' ( PDF ) . Inland Revenue ( now HMRC ) , and HM
Treasury . August 2002 . Archived from the original ( PDF ) on 6 February 2007 . Retrieved 17 May 2007 . `` Corporation tax reform -- A consultation document ''
( PDF ) . Inland Revenue ( now HMRC ) , and HM Treasury . August 2003 . Archived from the original ( PDF ) on 12 April 2007 . Retrieved 17 May 2007 . ``
Corporation tax reform -- Technical note '' ( PDF ) . Inland Revenue ( now HMRC ) . December 2004 . Archived from the original ( PDF ) on 21 February 2007 .
Retrieved 17 May 2007 . `` Key economic events 1955 -- 1979 '' .
</item 8>

<item 9>
Stand-your-ground law
The law was introduced in response to DPP v. Padraig Nally . A person who uses such force as is permitted by section 2 in the circumstances referred to in that
section shall not be liable in tort with respect to any injury , loss or damage arising from the use of such force . The force used is only such as is
reasonable in the circumstances as he or she believes them to be -- ( i ) to protect himself or herself or another person present in the dwelling from injury ,
assault , detention or death caused by a criminal act , ( ii ) to protect his or her property or the property of another person from appropriation , destruction
or damage caused by a criminal act , or ( iii ) to prevent the commission of a crime or to effect , or assist in effecting , a lawful arrest . It does not
matter whether the person using the force had a safe and practicable opportunity to retreat from the dwelling before using the force concerned . This law does
not apply to force used against a member of An Garda Siochána ( Irish Police ) or anyone assisting them , or a person lawfully performing a function authorised
by or under any enactment .
</item 9>

<item 10>
Indian High Courts Act 1861
1987 Finance Act ( India ) Fiscal Responsibility and Budget Management Act , 2003 Foreign Contribution Regulation Act Foreign Exchange Management Act
Geographical Indications of Goods ( Registration and Protection ) Act , 1999 Government Securities Act , 2006 Indian Contract Act , 1872 Indian Stamp Act , 1899
Insolvency and Bankruptcy Code , 2016 Insurance Act , 1938 Negotiable Instruments Act , 1881 Securities Laws ( Amendment ) Act , 2014 Securitisation and
Reconstruction of Financial Assets and Enforcement of Security Interest Act , 2002 The Competition Act , 2002 The High Denomination Bank Notes ( Demonetisation
) Act , 1978 The Income - tax Act , 1961 Transfer of Property Act 1882 Urban Land ( Ceiling and Regulation ) Act , 1976 Healthcare Central Council of
Homoeopathy Act , 1973 Clinical Establishments ( Registration and Regulation ) Act , 2010 Dentist Act Drugs and Cosmetics Act , 1940 Drugs and Magic Remedies (
Objectionable Advertisements ) Act , 1954 Indian Medical Council Act Mental Health Act , 1987 Narcotic Drugs and Psychotropic Substances Act ,
</item 10>
=== END ITEM id=16e1f06cce875bd7 ===

=== ITEM id=6d232558016777b6 ===
Gold sentences and facts:
Sentence 0: The soma ( cell body ) of each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root , where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons . The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
  Fact 0.0: The soma (cell body) of each pseudounipolar neuron is located within a dorsal root ganglion.
  Fact 0.1: The axon leaves the cell body.
  Fact 0.2: The axon leaves the dorsal root ganglion.
  Fact 0.3: The axon goes into the dorsal root.
  Fact 0.4: In the dorsal root, the axon splits into two branches.
  Fact 0.5: The central branch of the axon goes to the posterior (dorsal) horn of the spinal cord.
  Fact 0.6: The central branch forms synapses with other neurons.
  Fact 0.7: The peripheral branch travels through the distal dorsal root.
  Fact 0.8: The peripheral branch travels into the spinal nerve.
  Fact 0.9: The peripheral branch travels all the way until skin, joint, and muscle.

Retrieved items:
<item 1>
Pseudounipolar neuron
Ganglion Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion ( Preganglionic nerve fibers Postganglionic nerve fibers ) Connective tissues
Epineurium Perineurium Endoneurium Nerve fascicle Neuroglia Myelination : Schwann cell Neurilemma Myelin incisure Node of Ranvier Internodal segment Satellite
glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon Telodendron Axon terminals Axoplasm Axolemma Neurofibril / neurofilament Dendrite Nissl body
Dendritic spine Apical dendrite / Basal dendrite Types Bipolar Unipolar Pseudounipolar Multipolar Interneuron Renshaw Afferent nerve fiber / Sensory neuron GSA
GVA SSA SVA fibers Ia or Aα Ib or Golgi or Aα II or Aβ and Aγ III or Aδ or fast pain IV or C or slow pain Efferent nerve fiber / Motor neuron GSE GVE SVE Upper
motor neuron Lower motor neuron α motorneuron β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical
</item 1>

<item 2>
Motor neuron
These are found in the spinal cord and appear white from the myelin surrounding axons . While these tracts exist for both efferent and afferent nerves ,
efferent nerves will carry signals down the spinal cord toward their effector target . These tracts also serve as the place of origin for lower motor neurons .
There are seven major white matter motor tracts to be found in the spinal cord and are listed below : Lateral Corticospinal Tract Rubrospinal Tract Lateral
Reticulospinal Tract Vestibulospinal tract Medial Reticulospinal Tract Tectospinal Tract Anterior Corticospinal Tract Lower motor neurons ( edit ) Lower motor
neurons are those that originate in the spinal cord and directly or indirectly innervate effector targets . The target of these neurons varies , but in the
somatic nervous system the target will be some sort of muscle fiber . There are three primary categories lower motor neurons , which can be further divided in
sub-categories .
</item 2>

<item 3>
Pseudounipolar neuron
β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical synapse Synaptic vesicle Active zone Postsynaptic density Autapse
Ribbon synapse Neuromuscular junction Sensory receptors Meissner 's corpuscle Merkel nerve ending Pacinian corpuscle Ruffini ending Muscle spindle Free nerve
ending Nociceptor Olfactory receptor neuron Photoreceptor cell Hair cell Taste bud Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Pseudounipolar_neuron&oldid=833952921 '' Categories : Afferent neurons Hidden categories : Pages with unresolved
properties Talk Contents About Wikipedia Čeština Deutsch Français Italiano Svenska 中文 Edit links This page was last edited on 3 April 2018 , at 05 : 01 . About
Wikipedia
</item 3>

<item 4>
Pseudounipolar neuron
. Cambridge : McGrawHill . p. 24 . ISBN 978 - 0838577011 . ^ Jump up to : MacCallum , Don . `` Peripheral Nervous System '' . Histology and Virtual Microscopy
Learning Resources . University of Michigan Medical School . Archived from the original on 7 July 2014 . Retrieved 24 June 2014 . Jump up ^ Miller 's Anatomy of
the Dog , By Howard E. Evans , Alexander de Lahunta , 4th ed , pp 565 ( hide ) Nervous tissue CNS Tissue Types Grey matter White matter Projection fibers
Association fiber Commissural fiber Lemniscus Funiculus Fasciculus Nerve tract Decussation Commissure Neuropil Meninges Cell Types Neuronal Pyramidal Purkinje
Granule Glial insulating : Myelination : Oligodendrocyte other Astrocyte Radial glial cell Ependymal cells Tanycyte Microglia PNS General Dorsal Root Ganglion
Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion (
</item 4>

<item 5>
Pseudounipolar neuron
These axonal branches should not be confused with dendrites . These sensory neurons are an exception to the typical neuron , in that they do not have separate
dendrites and an axonal process , but rather one branched process that serves both functions . The axon has a peripheral branch ( from the cell body to the
periphery : skin , joint and muscle ) and a central branch ( from the cell body to spinal cord ) . In the dorsal root ganglia ( edit ) The soma ( cell body ) of
each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root
, where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons .
The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
</item 5>

<item 6>
Motor neuron
the command of visceral muscles is disynaptic ( involving two neurons : the general visceral motor neuron located in the CNS , which synapses onto a ganglionic
neuron , located in the PNS , which synapses onto the muscle ) . All vertebrate motor neurons are cholinergic , that is , they release the neurotransmitter
acetylcholine . Parasympathetic ganglionic neurons are also cholinergic , whereas most sympathetic ganglionic neurons are noradrenergic , that is , they release
the neurotransmitter noradrenaline . ( see Table ) Motor neurons in Regenerative medicine ( edit ) Human lower motor neurons can be generated in vitro from
embryonic stem cells and induced pluripotent stem cells . See also ( edit ) Betz cell Central chromatolysis Motor dysfunction Nerve References ( edit ) Jump up
^ Tortora , Gerard ; Derrickson , Bryan ( 2014 ) . Principles of Anatomy & Physiology ( 14th ed . ) . New Jersey : John Wiley & Sons , Inc . pp. 406 , 502 , 541
. ISBN 978 - 1 - 118 - 34500 - 9 .
</item 6>

<item 7>
Pseudounipolar neuron
Pseudounipolar neuron - wikipedia Pseudounipolar neuron Jump to : navigation , search Pseudounipolar neuron 1 : Pseudounipolar neuron , 2 : Bipolar neuron
Details System Nervous system Identifiers Latin neuron pseudounipolare TH H2. 00.06. 1.00047 Anatomical terms of neuroanatomy ( edit on Wikidata ) A
pseudounipolar neuron ( pseudo -- false , uni -- one ) is a kind of sensory neuron in the peripheral nervous system . This neuron contains an axon that has
split into two branches ; one branch runs to the periphery and the other to the spinal cord . Contents ( hide ) 1 Structure 1.1 In the dorsal root ganglia 1.2
In the sensory ganglia of cranial nerves 2 Function 2.1 Signal pathway 3 See also 4 References Structure ( edit ) By definition , a pseudounipolar neuron has
one axon with two branches : central and peripheral . These axonal branches should not be confused with dendrites .
</item 7>

<item 8>
Motor neuron
segment Satellite glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon Telodendron Axon terminals Axoplasm Axolemma Neurofibril / neurofilament
Dendrite Nissl body Dendritic spine Apical dendrite / Basal dendrite Types Bipolar Unipolar Pseudounipolar Multipolar Interneuron Renshaw Afferent nerve fiber /
Sensory neuron GSA GVA SSA SVA fibers Ia or Aα Ib or Golgi or Aα II or Aβ and Aγ III or Aδ or fast pain IV or C or slow pain Efferent nerve fiber / Motor neuron
GSE GVE SVE Upper motor neuron Lower motor neuron α motorneuron β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical
synapse Synaptic vesicle Active zone Postsynaptic density Autapse Ribbon synapse Neuromuscular junction Sensory receptors Meissner 's corpuscle Merkel nerve
ending Pacinian corpuscle Ruffini ending Muscle spindle Free nerve ending Nociceptor Olfactory receptor neuron Photoreceptor cell Hair cell Taste bud Retrieved
from `` https://en.wikipedia.
</item 8>

<item 9>
Pseudounipolar neuron
In the sensory ganglia of cranial nerves ( edit ) Pseudounipolar neurons are also found in the sensory ganglia of cranial nerves V , VII , IX and X , all mixed
- cranial nerves . The vestibulocochlear nerve , has the bipolar neuron type in both the spiral ganglion as well as Scarpa 's ganglion . Function ( edit )
Signal pathway ( edit ) These neurons have sensory receptors on skin , joints , muscles , and other parts of the body . The area of the axon that is closest to
the receptor is the trigger zone for the neuron . The signal is conducted through the axon to the dorsal root ganglion 's cell body , then through the axon and
ending at the sensory nuclei in the posterior horn of the spinal cord . See also ( edit ) Bipolar neuron Multipolar neuron Unipolar neuron References ( edit ) ^
Jump up to : Kandel , Eric R. ; Schwartz , James H. ; Mack , Thomas M. Jessell ( 2000 ) . Principles of neural Science ( 4th ed . ) . Cambridge : McGrawHill .
p. 24 .
</item 9>

<item 10>
Motor neuron
These columns are further explained in the table below . Motor Columns of Spinal Cord Motor Column Location in Spinal Cord Target Median Motor Column Present
entire length Axial muscles Hypaxial Motor Column Thoracic Region Body wall muscles Preganglionic Motor Column Thoracic Region Sympathetic ganglion Laterl Motor
Column Brachial and Lumbar region ( both regions are further divided into medial and lateral domains ) Muscles of the limbs Phrenic Motor Coulmn Cervical Region
Diaphragm Anatomy and Physiology ( edit ) Spinal cord tracts Location of lower motor neurons in spinal cord Upper motor neurons ( edit ) Upper motor neurons
originate in the motor cortex located in the precentral gyrus . The cells that make up the primary motor cortex are Betz cells , which are a type of pyramidal
cell . The axons of these cells descend from the cortex to form the corticospinal tract . White matter Tract ( edit ) White matter tracts are bundles of axons
and serve as means to propagate nerve impulses on a large scale . These are found in the spinal cord and appear white from the myelin surrounding axons .
</item 10>
=== END ITEM id=6d232558016777b6 ===
