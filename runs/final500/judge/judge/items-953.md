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

=== ITEM id=eec89a29b3858b79 ===
Gold sentences and facts:
Sentence 0: Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double Down
  Fact 0.0: The title is Diary of a Wimpy Kid: The Getaway.
  Fact 0.1: The author is Jeff Kinney.
  Fact 0.2: The country is United States.
  Fact 0.3: The language is English.
  Fact 0.4: The series is Diary of a Wimpy Kid.
  Fact 0.5: The genres are children's novel and graphic novel.
  Fact 0.6: The publisher is Amulet Books (US).
  Fact 0.7: The publisher is Puffin Books (UK).
  Fact 0.8: The publication date is November 7, 2017 (worldwide).
  Fact 0.9: The media type is print (hardcover).
  Fact 0.10: The book has 217 story pages, 224 altogether.
  Fact 0.11: The ISBN is 978-1-4197-2545-6.
  Fact 0.12: It is preceded by Double Down.

Retrieved items:
<item 1>
Graph query row:
title: Diary of a Wimpy Kid: The Getaway
genre: Graphic Novel
genre_kind: genre
relationship: RELATED_TO
</item 1>

<item 2>
Diary of a Wimpy Kid: The Getaway
Please discuss this issue on the article 's talk page . ( December 2017 ) This article needs more links to other articles to help integrate it into the
encyclopedia . Please help improve this article by adding links that are relevant to the context within the existing text . ( December 2017 ) ( Learn how and
when to remove this template message ) ( Learn how and when to remove this template message ) Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country
United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication
date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double
Down Diary of a Wimpy Kid : The Getaway is the twelfth book in the Diary of a Wimpy Kid series by Jeff Kinney .
</item 2>

<item 3>
Graph query row:
title: Diary of a Wimpy Kid: The Getaway
genre: Children's novel
genre_kind: genre
relationship: RELATED_TO
</item 3>

<item 4>
Diary of a Wimpy Kid: The Getaway
The couple sit on either side of him with their baby . Greg becomes paranoid when he watches the plane 's safety video and begins to ask questions about
unfortunate events . Greg is very aggravated of the other passengers because most are n't watching the safety video . Mid flight , he feels boxed in when the
people in front recline and the person behind slides his feet on Greg 's armrest . Greg tries to recline his chair too but when he asks the flight attendant how
, he learns that he ca n't because the row behind him is an emergency exit . He tries to swap seats with Frank but Rodrick beats him to it . He stays in the
bathroom because the couple 's baby started to get annoying , but then he returns to his seat after experiencing turbulence . He somehow was able to sleep for
the rest of the flight . As they land on Christmas Day , Greg begins to understand why the family came there . They touch down and began to head to the resort ,
Isla de Corales . He is even excited by the activities the resort offer which he sees on the bus .
</item 4>

<item 5>
Diary of a Wimpy Kid: The Getaway
Waterstones.com . Retrieved 10 December 2017 . ( hide ) The Diary of a Wimpy Kid series by Jeff Kinney Novels Diary of a Wimpy Kid Rodrick Rules The Last Straw
Dog Days The Ugly Truth Cabin Fever The Third Wheel Hard Luck The Long Haul Old School Double Down The Getaway Characters Greg Heffley Film series Diary of a
Wimpy Kid Rodrick Rules Dog Days The Long Haul Retrieved from `` https://en.wikipedia.org/w/index.
</item 5>

<item 6>
Diary of a Wimpy Kid: The Getaway
Diary of a Wimpy Kid : the Getaway - wikipedia Diary of a Wimpy Kid : the Getaway Jump to : navigation , search ( hide ) This article has multiple issues .
Please help improve it or discuss these issues on the talk page . ( Learn how and when to remove these template messages ) This article consists almost entirely
of a plot summary . It should be expanded to provide more balanced coverage that includes real - world context . Please edit the article to focus on discussing
the work rather than merely reiterating the plot . ( August 2016 ) ( Learn how and when to remove this template message ) This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
December 2017 ) ( Learn how and when to remove this template message ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page .
</item 6>

<item 7>
Diary of a Wimpy Kid: The Getaway
Release date ( edit ) The Getaway is the 12th book in the series , and was released everywhere except Canada and China on November 7 , 2017 . Reception ( edit )
A week since its release , it has received 4.5 / 5 stars on Waterstones with 786 reviews . References ( edit ) Jump up ^ Wimpy Kid ( 27 April 2017 ) . `` Wimpy
Kid LIVE Special 10th Anniversary Presentation and Cover Reveal of Book 12 ! '' . YouTube . Retrieved 10 December 2017 . Jump up ^ `` Cover Of New Diary Of A
Wimpy Kid Book Revealed By Jeff Kinney To Fans Worldwide At Live streamed Carnegie Hall Event Hosted By ABRAMS '' . Tristateupdate.com . Retrieved 10 December
2017 . Jump up ^ `` Diary of a Wimpy Kid : The Getaway by Jeff Kinney -- Waterstones '' . Waterstones.com . Retrieved 10 December 2017 .
</item 7>

<item 8>
Diary of a Wimpy Kid: The Getaway
They enter the line for security , but a child unclips a barrier separating the lines , and everybody gets scrambled . They go through security and are delayed
again by somebody who has to remove all of his piercings . Manny then gets his toy stuck in an escalator , delaying them more . As a result , the family panic ,
grab some luggage and dash to the gate . It turns out their flight is delayed . Greg goes to dry his sock but destroys it while doing so . He hears that the
flight is overbooked , and tries to stop his family boarding in exchange of a hotel stay and $300 ( soon $500 ) , but is stopped by Susan . A replacement crew
for the plane is required and they finally board . Greg is first impressed by the plane 's first class seats before he discovers that he has to sit in economy .
However , Frank collected enough air miles because of his business trips to sit in first class . In economy , Mom , Rodrick , and Manny are seated in the right
row . Greg finds out he has three seats to himself , until a couple with a baby arrives late on board .
</item 8>

<item 9>
Diary of a Wimpy Kid: The Getaway
The book was unveiled during the 2017 Diary of a Wimpy Kid Virtually Live Event which was live streamed via YouTube as part of the 10th anniversary of the first
book . The book was published and released on November 7 , 2017 . Contents ( hide ) 1 Plot 2 Release date 3 Reception 4 References Plot ( edit ) After many
frustrations , Greg Heffley 's parents decide to skip Christmas and travel by plane to Isla De Corales , the hotel they stayed at while on their honeymoon much
to Greg 's dismay . On Christmas Eve they drive to the airport , but are delayed by a series of mishaps . They encounter holiday traffic but arrive at the
airport . As they fail to unload in time at the drop off zone , Frank and Greg have to return later with the luggage . Greg steps in a slush puddle and they end
up missing the shuttle . In the airport , the family 's suitcases are too heavy , so they stuff some clothes into their carry on baggage .
</item 9>

<item 10>
Diary of a Wimpy Kid: The Getaway
He is even excited by the activities the resort offer which he sees on the bus . However , Frank and Susan are disappointed that the resort has since changed ,
and it is now split with a family side ( the Mild Side ) and a couple side ( the Wild Side ) . The family also discovers that they got the wrong luggage that is
locked , and their luggage was sent back home . They only have the clothes they put in their carry on luggage . They attempt to buy some new clothes , but Frank
wo n't buy them as they are too expensive . The family goes to explore the resort and end up at the pool . The Director of Fun shows up , however , and gets
everybody moving . Greg is forced to take part in two games . After this , Greg sees an iguana by the pool and is worried , so the family goes to get some lunch
. Greg and his family try to eat outside ; however , too many birds and bugs keep ruining their meals . Frank contracts a stomach bug from his water , so they
head back to the suite . The next day , Greg uses the shower which is outside .
</item 10>
=== END ITEM id=eec89a29b3858b79 ===

=== ITEM id=2598e4ec0205647d ===
Gold sentences and facts:
Sentence 0: The previous department issued firearm for CHP officers is the Smith & Wesson Model 4006 TSW in . 40 S&W . As of August 2017 , the department has begun replacing the 4006 TSW with the Smith & Wesson M&P in . 40 S&W . Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12 ‐ gauge shotgun and a Sig Sauer M400 rifle in 5.56 mm . Less Lethal options for officers include OC Pepper Spray and the PR - 24 Side Handle Baton . The ASP Expandable Straight Baton is authorized as an optional piece of equipment for officers completing their probationary year . Additionally , officers are authorized to carry the TASER X2 Defender . In early 2009 , officers have been allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols .
  Fact 0.0: The previous department-issued firearm for CHP officers is the Smith & Wesson Model 4006 TSW in .40 S&W.
  Fact 0.1: As of August 2017, the CHP department has begun replacing the Smith & Wesson 4006 TSW with the Smith & Wesson M&P in .40 S&W.
  Fact 0.2: Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12-gauge shotgun.
  Fact 0.3: Each CHP patrol vehicle is equipped with a Sig Sauer M400 rifle in 5.56 mm.
  Fact 0.4: Less lethal options for CHP officers include OC Pepper Spray.
  Fact 0.5: Less lethal options for CHP officers include the PR-24 Side Handle Baton.
  Fact 0.6: The ASP Expandable Straight Baton is authorized as an optional piece of equipment for CHP officers completing their probationary year.
  Fact 0.7: CHP officers are authorized to carry the TASER X2 Defender.
  Fact 0.8: In early 2009, CHP officers were allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols.

Retrieved items:
<item 1>
The California Highway Patrol (CHP) is California’s statewide highway patrol and law-enforcement agency, established August 14, 1929. Its motto is “Safety,
Service, and Security.” In 2014, it reported 10,551 employees and a $2 billion annual budget. Its jurisdiction spans California’s 163,696 square miles and
serves a population estimated at 39.5 million in 2017.
</item 1>

<item 2>
California Highway Patrol
This is because of their ability to carry a larger `` basic load '' of normal issue equipment , equipment that may be only in a few cars , and / or not carried
by patrol cars at all , in a more urban setting . Aircraft ( edit ) CHP AS 350 B3 in flight . The CHP utilizes both helicopters and fixed - wing aircraft . In
2017 the fleet consisted of : 15 Airbus Helicopters ( Eurocopter ) AS350B3 Squirrels 11 Cessna T206H Turbo Stationairs 4 Gippsaero GA8 TC - 320 Airvans 2 Hughes
( MD500 ) 369Ds 1 Cessna 182S Skylane 1 Raytheon ( Beechcraft ) B300 Super King Air Rogue officers ( edit ) George Gwaltney ( edit ) A former officer from
Barstow , was convicted on May 10 , 1984 , in federal court , for the on ‐ duty rape and slaying of 23 - year - old Robin Bishop on January 11 , 1982 .
</item 2>

<item 3>
Established in 1929, the California Highway Patrol is California’s statewide law-enforcement agency, with over 10,500 employees and jurisdiction across the
state. It patrols highways and county roads, enforces traffic laws, investigates crashes, inspects commercial vehicles and school buses, protects state
officials and facilities, and supports emergencies and local agencies. Its divisions oversee 103 area offices. CHP uses cars, trucks, motorcycles, aircraft, and
K-9 teams; it also runs safety programs and
</item 3>

<item 4>
California Highway Patrol
Weapons ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced
material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) The previous department issued firearm for
CHP officers is the Smith & Wesson Model 4006 TSW in . 40 S&W . As of August 2017 , the department has begun replacing the 4006 TSW with the Smith & Wesson M&P
in . 40 S&W . Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12 ‐ gauge shotgun and a Sig Sauer M400 rifle in 5.56 mm . Less Lethal
options for officers include OC Pepper Spray and the PR - 24 Side Handle Baton . The ASP Expandable Straight Baton is authorized as an optional piece of
equipment for officers completing their probationary year . Additionally , officers are authorized to carry the TASER X2 Defender . In early 2009 , officers
have been allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols .
</item 4>

<item 5>
California Highway Patrol
Cold weather and utility uniforms are dark blue . California Highway Patrol campaign hat . Leather gear is black with brass snaps and buckles . Standard traffic
enforcement patrol vehicles are required by state law to have a white door with , in the case of the CHP , a star . The CHP operates traditional black and white
as well as all - white patrol vehicles . The California Highway Patrol is one of the few organizations to continue to use the older toll - free `` Zenith 1 ‐
2000 '' number . With the falling cost of telephone area code 800 and 888 numbers , most organizations have chosen to switch to one of the newer numbers and
discontinue use of the Zenith service , which requires operator assistance . The CHP 's traditions include its own radio codes , widely adopted by local
agencies . The most important is 11 ‐ 99 , which signifies that an officer needs emergency assistance or that an officer is down . In 1981 a charitable
foundation called the 11 ‐ 99 Foundation was founded to provide benefits and scholarships to officers and their families .
</item 5>

<item 6>
The California Highway Patrol (CHP) patrols California’s state, U.S. and interstate highways, plus public roads in both incorporated and unincorporated county
areas. Its traditions include dark-blue uniforms, black leather gear, marked patrol cars, distinctive radio codes and the legacy Zenith 1-2000 phone number. The
code “11-99” signals an officer needs emergency help; a foundation established in 1981 supports officers and their families. CHP officers pledge to serve
honorably
</item 6>

<item 7>
California Highway Patrol
ESPN March 6 2008 Jump up ^ Game - warden shortage is about to get worse San Francisco Chronicle September 23 2007 Archived February 15 , 2012 , at the Wayback
Machine . Jump up ^ Lots of ocean , but few game wardens ! Sacramento Bee August 23 2007 Archived August 21 , 2007 , at the Wayback Machine . Jump up ^ Oregon
State Police Fish and Wildlife Division Archived 2012 - 07 - 31 at the Wayback Machine . Jump up ^ Division of Alaska Wildlife Troopers Jump up ^
http://smith-wessonforum.com/showthread.php?t=525162 ^ Jump up to : Mattos , Rick ( 2008 ) . California Highway Patrol . Charleston , SC : Arcadia Pub . ISBN
9780738556208 . Retrieved 2013 - 09 - 01 . Jump up ^ Carpenter , Susan ( 25 June 2013 ) . `` BMW 's R1200RT - P : Big , bad and packing heat '' . Los Angeles
Times . Retrieved 1 September 2013 .
</item 7>

<item 8>
California Highway Patrol
Early motorcycles used included Indian , Harley - Davidson and Henderson manufactured bikes during the 1920s and 30s ; though by 1941 the main manufacturers
used at the training academy were Harley - Davidson and Indian . In the 1970s CHP decided to renew their Harleys fleet and tested many motorcycles such as Honda
750 four , Suzuki 850 , Moto Guzzi and finally Kawasaki Z900 and then 1000C1 made famous in CHiPs . In 1998 CHP began using the BMW R1100RT - P , with features
including a helmet interface for communications , two batteries , and a top speed of 130 miles per hour ( 210 km / h ) . Vehicles ( edit ) California Highway
Patrol cruiser on display at Public Safety Day in Lakewood , California . The Department of General Services is responsible for selecting a vehicle based on
price , performance and payload capacity for the CHP . The bid specifications require a pursuit - rated , rear - wheel drive or all - wheel drive that can carry
at least 1,500 pounds , the approximate weight of four officers , their equipment , and the police vehicle equipment .
</item 8>

<item 9>
California Highway Patrol
Legal jurisdiction As per operations jurisdiction Governing body California State Legislature Constituting instrument Senate Bill 869 Headquarters Sacramento ,
California Officers 7,608 ( as of 2016 ) Civilians 3,276 ( as of 2014 ) Commissioner responsible Warren Stanley Parent agency California State Transportation
Agency ( CALSTA ) Facilities Commands 8 Field Divisions 16 Commercial Enforcement Facilities 103 Area Offices Patrol cars Ford Explorer - based Police
Interceptor Utility Vehicle Dodge Charger - Police Ford Crown Victoria - Police Planes Cessna 206 Website http://www.chp.ca.gov http://www.chpcareers.com The
California Highway Patrol ( CHP ) is a law enforcement agency of the state of California . The primary mission of the CHP is to ensure safety and enforce
traffic laws on all California highways and county roads in unincorporated areas . The CHP promotes traffic safety by inspecting commercial vehicles , as well
as inspecting and certifying school buses , ambulances , and other specialized vehicles . The California State Legislature established the California Highway
Patrol as a branch of the Division of Motor Vehicles in the Department of Public Works , with legislation signed by Governor C. C. Young on August 14 , 1929 .
</item 9>

<item 10>
California Highway Patrol
Vehicles ( edit ) This section needs expansion . You can help by adding to it . ( September 2013 ) Origins ( edit ) When motor vehicles in California were first
seen as needing legislation , law enforcement agencies began to patrol using motorcycles , cars and trucks . Motorcycle officers in 1920 Fresno , started a
group to assist each other and promote road safety - the Joaquin Valley Traffic Officer 's Association - led by Harry Wilson who they elected as president .
They renamed themselves the California Association of Highway Patrolmen in 1921 , and became the California Highway Patrol in 1927 under the auspices of the
Department of Motor Vehicles . Motorcycles ( edit ) A motorcycle sergeant on the San Tomas Expressway near Benton Avenue in Santa Clara , California . Through
the public competitive bidding process , the Harley - Davidson motorcycle was selected as the primary enforcement motorcycle for the California Highway Patrol
in 2013 . These replacement enforcement motorcycles will replenish the Department 's aging motorcycle fleet . In a cost - saving move , the CHP previously
deferred the purchase of replacement motorcycles and has not purchased enforcement motorcycles since January 14 , 2011 .
</item 10>
=== END ITEM id=2598e4ec0205647d ===

=== ITEM id=16ba7792ddbc6fa2 ===
Gold sentences and facts:
Sentence 0: Top 20 rankings as of 16 October 2017 Rank Change Team Points Germany 1631 Brazil 1619 Portugal 1446 Argentina 1445 5 Belgium 1333 6 Poland 1323 7 France 1226 8 Spain 1218 9 Chile 1173 10 Peru 1160 11 Switzerland 1134 12 England 1116 13 Colombia 1095 14 Wales 1072 15 Italy 1066 16 Mexico 1060 17 Uruguay 1034 18 Croatia 1013 19 7 Denmark 1001 20 9 Netherlands 931 * Change from 14 September 2017 Complete rankings at FIFA.com
  Fact 0.0: The top 20 rankings are as of 16 October 2017.
  Fact 0.1: Germany is ranked 1 with 1631 points.
  Fact 0.2: Brazil is ranked 2 with 1619 points.
  Fact 0.3: Portugal is ranked 3 with 1446 points.
  Fact 0.4: Argentina is ranked 4 with 1445 points.
  Fact 0.5: Belgium is ranked 5 with 1333 points.
  Fact 0.6: Poland is ranked 6 with 1323 points.
  Fact 0.7: France is ranked 7 with 1226 points.
  Fact 0.8: Spain is ranked 8 with 1218 points.
  Fact 0.9: Chile is ranked 9 with 1173 points.
  Fact 0.10: Peru is ranked 10 with 1160 points.
  Fact 0.11: Switzerland is ranked 11 with 1134 points.
  Fact 0.12: England is ranked 12 with 1116 points.
  Fact 0.13: Colombia is ranked 13 with 1095 points.
  Fact 0.14: Wales is ranked 14 with 1072 points.
  Fact 0.15: Italy is ranked 15 with 1066 points.
  Fact 0.16: Mexico is ranked 16 with 1060 points.
  Fact 0.17: Uruguay is ranked 17 with 1034 points.
  Fact 0.18: Croatia is ranked 18 with 1013 points.
  Fact 0.19: Denmark is ranked 19 with 1001 points and has a rank change of 7.
  Fact 0.20: Netherlands is ranked 20 with 931 points and has a rank change of 9.
  Fact 0.21: The rank changes are measured from 14 September 2017.
  Fact 0.22: The complete rankings are available at FIFA.com.

Retrieved items:
<item 1>
FIFA World Rankings
Retrieved 13 February 2009 . So the news this week that Israel has risen another place to number 15 in the latest FIFA rankings , above Mexico ( 25 ) , Nigeria
( 22 ) , the US ( 24 ) and Colombia ( 40 ) , only serves to underline the futility of the ranking system . Jump up ^ `` FIFA Ranking : November 2008 preview (
II ) '' . Football-Rankings.info. 20 October 2008 . Retrieved 13 February 2009 . Israel will improve ( again ! ) their best ever ranking , climbing one place to
15th . With a win ( instead of a draw ) against Latvia they would have been tied for 8th place with Russia . Jump up ^ Alistair Tweedale ( 13 October 2015 ) .
`` Belgium rise to No1 in Fifa world rankings after they beat Israel -- despite playing one tournament in 13 years '' . The Telegraph . Retrieved 14 November
2015 . Jump up ^ `` FIFA Rankings '' . Travour.com. 1 June 2006 .
</item 1>

<item 2>
FIFA World Rankings
Archived from the original on 25 December 2011 . Retrieved 13 April 2011 . Jump up ^ `` Olympics -- updated draw procedure , with teams '' . CAFOnline.com .
Confederation of African Football . 12 April 2011 . Retrieved 13 April 2011 . ^ Jump up to : `` The world rankings riddle '' . BBC Sport. 21 December 2000 .
Jump up ^ `` FIFA adapting new world rankings '' ( PDF ) . Associated Press . 2 June 2006 . Retrieved 23 July 2017 . Jump up ^ `` FIFA chief praises IFA 's work
'' . The Jerusalem Post . 29 August 2008 . Archived from the original on 27 April 2011 . Retrieved 13 February 2009 . Blatter was also asked about Israel 's
unrealistic FIFA ranking . Despite failing to qualify for a major tournament in 38 years , the national team is currently ranked number 16 in the world , ahead
of such team as Greece , Sweden and Denmark , and just two places below England .
</item 2>

<item 3>
FIFA World Rankings
FIFA World rankings - wikipedia FIFA World rankings This is the latest accepted revision , reviewed on 25 October 2017 . Jump to : navigation , search This
article is about the men 's rankings . For the women 's rankings , see FIFA Women 's World Rankings . Top 20 rankings as of 16 October 2017 Rank Change Team
Points Germany 1631 Brazil 1619 Portugal 1446 Argentina 1445 5 Belgium 1333 6 Poland 1323 7 France 1226 8 Spain 1218 9 Chile 1173 10 Peru 1160 11 Switzerland
1134 12 England 1116 13 Colombia 1095 14 Wales 1072 15 Italy 1066 16 Mexico 1060 17 Uruguay 1034 18 Croatia 1013 19 7 Denmark 1001 20 9 Netherlands 931 * Change
from 14 September 2017 Complete rankings at FIFA.com The FIFA World Ranking is a ranking system for men 's national teams in association football , currently
led by Germany .
</item 3>

<item 4>
FIFA World Rankings
The perceived flaws in the FIFA system have led to the creation of a number of alternative rankings from football statisticians , including the World Football
Elo Ratings and the rec. sport. soccer Statistics Foundation rankings . The current system has been criticised for being open to abuse . National teams can
maintain a higher average score by playing fewer friendly matches , particularly against weaker opponents . For example , Switzerland were seeded for the 2014
World Cup draw , largely thanks to only playing three friendly matches in the previous year . Switzerland would have reduced their rating if they had played any
more friendlies , as their average score was greater than the maximum possible score from winning a friendly match against the top - ranked nation . Romania
were one of the surprise seeds for the 2018 World Cup preliminary draw after taking advice from a consultant and playing only one friendly in the year before
the draw . Lastly , the use of regional strength multiplier in the ranking determination formula has been accused of further reinforcing and perpetuating the
bias for and against certain regions . FIFA 's use of regional multiplier has been cited as a primary reason why most teams in AFC and CONCACAF have
significantly lower FIFA rankings compared to their Elo rating based rankings .
</item 4>

<item 5>
FIFA World Rankings
Jump up ^ `` Belgium and Turkey claim awards , Hungary return '' . fifa.com. 3 December 2015 . Archived from the original on 21 July 2016 . Retrieved 15
December 2015 . Jump up ^ `` Argentina and France take Ranking awards '' . fifa.com. 22 December 2016 . Archived from the original on 22 December 2016 .
Retrieved 12 December 2016 . Jump up ^ Men 's Ranking Procedure on the FIFA website External links ( edit ) FIFA World Ranking FIFA rankings : A Guide to the
very first ' 93 revision ( RSSSF ) How the FIFA men 's rankings are calculated Interactive world map of the FIFA rankings FIFA History of FIFA FIFA Anthem FIFA
Congress FIFA Council FIFA Ethics Committee FIFA headquarters Football at the Summer Olympics List of football federations International Football Association
Board Timeline of association football Football codes Association football Beach soccer Futsal Confederations AFC CAF CONCACAF CONMEBOL OFC UEFA Men 's
tournaments FIFA World Cup FIFA Confederations Cup FIFA U-20 World Cup FIFA U-17 World Cup FIFA Club World Cup FIFA Futsal World Cup FIFA Beach Soccer World Cup
Blue Stars /
</item 5>

<item 6>
FIFA World Rankings
Jump up ^ `` Soccer : Quirky FIFA rankings system sees Israel climb to # 15 '' . The Jerusalem Post . 13 November 2008 . Archived from the original on 16
November 2008 . Retrieved 13 February 2009 . Israel continued its baffling rise up the FIFA rankings on Wednesday , climbing one more position to a best - ever
15th in the world . Amazingly , the national team , which has n't reached a major tournament since the 1970 World Cup , is ahead of such teams as Greece ( 18 )
, Nigeria ( 22 ) , Sweden ( 29 ) , Scotland ( 33 ) , Denmark ( 34 ) and the Republic of Ireland ( 36 ) Jump up ^ `` The Last Word : It 's time to get rid of the
FIFA rankings '' . The Jerusalem Post . 14 November 2008 . Archived from the original on 16 November 2008 . Retrieved 13 February 2009 .
</item 6>

<item 7>
FIFA World Rankings
Travour.com. 1 June 2006 . Archived from the original on 14 June 2006 . Jump up ^ Cummings , Michael . `` FIFA World Rankings Place Brazil 18th , Reinforce
Flawed Nature of System '' . Retrieved 21 May 2013 . Jump up ^ Freitas , Paulo . `` Scolari blames FIFA for Brazil 's poor ranking '' . Retrieved 16 June 2013 .
^ Jump up to : Burnton , Simon ( 17 October 2013 ) . `` With a bit of planning England could have been seeded ahead of the Swiss '' . The Guardian . Retrieved
15 June 2015 . Jump up ^ `` Romania , Wales set to be surprise World Cup top seeds '' . USA Today . Associated Press . 23 July 2015 . Retrieved 29 July 2015 .
Romania took advice from a consultant who decoded some mysteries of a ranking system used by FIFA since 1992 . Jump up ^ Gavril , Victor ( 5 July 2015 ) .
</item 7>

<item 8>
FIFA World Rankings
The teams of the member nations of FIFA , football 's world governing body , are ranked based on their game results with the most successful teams being ranked
highest . The rankings were introduced in December 1992 , and eight teams ( Argentina , Belgium , Brazil , France , Germany , Italy , the Netherlands and Spain
) have held the top position , of which Brazil have spent longest ranked first . A points system is used , with points being awarded based on the results of all
FIFA - recognised full international matches . Under the existing system , rankings are based on a team 's performance over the last four years , with more
recent results and more significant matches being more heavily weighted to help reflect the current competitive state of a team . The ranking system was most
recently revamped after the 2006 World Cup , with the first edition of the new series of rankings issued on 12 July 2006 . The most significant change is that
the rankings are now based on results over the previous four years instead of the previous eight years .
</item 8>

<item 9>
FIFA World Rankings
FIFA Puskás Award FIFA Women 's World Cup awards FIFA World Coach of the Year FIFA World Cup All - Time Team FIFA World Cup Dream Team FIFA World Cup awards
FIFA World Player of the Year The Best FIFA Football Awards Rankings FIFA World Rankings FIFA World Ranking system ( 1999 -- 2006 ) FIFA Women 's World Rankings
Congresses 51st ( Paris 1998 ) 53rd ( Seoul 2002 ) 61st ( Zürich 2011 ) 65th ( Zürich 2015 ) Extraordinary ( Zürich 2016 ) Corruption `` FIFA 's Dirty Secrets
'' Garcia Report 2015 FIFA corruption case List of banned football officials Others FIFA ( video game series ) List of FIFA country codes FIFA Disciplinary Code
FIFA Fan Fest FIFA Futbol Mundial FIFA eligibility rules FIFA International Match Calendar FIFA International Referees List FIFA Master FIFA Transfer Matching
System FIFA World Cup Trophy Non-FIFA United Passions International association football FIFA Federations Teams Competitions World Cup U-17 U-20 Confederations
Cup Olympics Youth Olympics Universiade World Rankings The Best FIFA Football Awards Timeline of association football Comparison of association football and
futsal Africa CAF -- Africa Cup of Nations U-23
</item 9>

<item 10>
FIFA World Rankings
In July 2014 , Germany took over the lead once again , having won the 2014 FIFA World Cup . In July 2015 , Argentina reached the top spot for the first time
since 2008 , after reaching both the 2014 FIFA World Cup Final , as well as the 2015 Copa America Final . In November 2015 , Belgium became the leader in the
FIFA rankings for the first time , after topping their Euro 2016 qualifying group . Belgium led the rankings until April 2016 , when Argentina returned to the
top . On April 6 , 2017 , Brazil returned to the No. 1 spot for the first time since just prior to the 2010 World Cup , but Germany regained the top spot in
July after winning the Confederations Cup . Uses of the rankings ( edit ) The rankings are used by FIFA to rank the progression and current ability of the
national football teams of its member nations , and claims that they create `` a reliable measure for comparing national A-teams '' . They are used as part of
the calculation , or the entire grounds to seed competitions .
</item 10>
=== END ITEM id=16ba7792ddbc6fa2 ===

=== ITEM id=18c591f2ac71c849 ===
Gold sentences and facts:
Sentence 0: As a technique for drawing , the closest predecesor to the pencil was Silverpoint until in 1565 ( some sources say as early as 1500 ) , a large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite in Borrowdale parish , Cumbria , England . This particular deposit of graphite was extremely pure and solid , and it could easily be sawn into sticks . It remains the only large - scale deposit of graphite ever found in this solid form . Chemistry was in its infancy and the substance was thought to be a form of lead . Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally mean lead pen .
  Fact 0.0: Before the pencil, the closest predecessor as a drawing technique was Silverpoint, until 1565 (some sources say as early as 1500).
  Fact 0.1: A large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite.
  Fact 0.2: The Grey Knotts graphite deposit is in Borrowdale parish, Cumbria, England.
  Fact 0.3: The graphite deposit at Grey Knotts was extremely pure and solid.
  Fact 0.4: The graphite deposit at Grey Knotts could easily be sawn into sticks.
  Fact 0.5: The Grey Knotts graphite deposit remains the only large-scale deposit of graphite ever found in this solid form.
  Fact 0.6: Chemistry was in its infancy when the graphite was found.
  Fact 0.7: The substance was thought to be a form of lead.
  Fact 0.8: The substance was called plumbago, Latin for 'lead ore'.
  Fact 0.9: The pencil core is still referred to as 'lead' or 'a lead'.
  Fact 0.10: Many people have the misconception that the graphite in the pencil is lead.
  Fact 0.11: The black core of pencils is still referred to as lead.
  Fact 0.12: The black core of pencils never contained the element lead.
  Fact 0.13: The German word for pencil, bleistift, literally means lead pen.
  Fact 0.14: The Irish word for pencil, peann luaidhe, literally means lead pen.
  Fact 0.15: The Arabic word for pencil, قلم رصاص (qalam raṣāṣ), literally means lead pen.
  Fact 0.16: Some other languages also use words for pencil that literally mean lead pen.

Retrieved items:
<item 1>
Pencil
Because graphite is soft , it requires some form of encasement . Graphite sticks were initially wrapped in string or sheepskin for stability . England would
enjoy a monopoly on the production of pencils until a method of reconstituting the graphite powder was found in 1662 in Italy . However , the distinctively
square English pencils continued to be made with sticks cut from natural graphite into the 1860s . The town of Keswick , near the original findings of block
graphite , still manufactures pencils , the factory also being the location of the Cumberland Pencil Museum . The meaning of `` graphite writing implement ''
apparently evolved late in the 16th century . Wood holders added Palomino Blackwing 602 pencils Around 1560 , an Italian couple named Simonio and Lyndiana
Bernacotti made what are likely the first blueprints for the modern , wood - encased carpentry pencil . Their version was a flat , oval , more compact type of
pencil . Their concept involved the hollowing out of a stick of juniper wood .
</item 1>

<item 2>
Pencil
Archived from the original on 5 March 2016 . Retrieved 5 October 2015 . Jump up ^ Bennett , Howard J. ( 26 November 2014 ) . `` Ever wondered about the lead in
pencils ? '' . The Washington Post . ISSN 0190 - 8286 . Archived from the original on 6 November 2015 . Retrieved 5 October 2015 . Jump up ^ `` Pencil
swallowing : MedlinePlus Medical Encyclopedia '' . www.nlm.nih.gov . Archived from the original on 6 October 2015 . Retrieved 5 October 2015 . Jump up ^ ``
graphite pencils The Weekend Historian '' . umeshmadan.wordpress.com . Archived from the original on 7 October 2015 . Retrieved 5 October 2015 . Jump up ^ ``
Origins of the Pencil ( Circa 1500 -- 1565 ) : HistoryofInformation.com '' . www.historyofinformation.com . Retrieved 2016 - 10 - 29 .
</item 2>

<item 3>
Pencil
Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people
have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the
element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally
mean lead pen . The value of graphite would soon be realised to be enormous , mainly because it could be used to line the moulds for cannonballs ; the mines
were taken over by the Crown and were guarded . When sufficient stores of graphite had been accumulated , the mines were flooded to prevent theft until more was
required . The usefulness of graphite for pencils was discovered as well , but graphite for pencils had to be smuggled . The news of the usefulness of these
early pencils spread far and wide , attracting the attention of artists all over the known world . Because graphite is soft , it requires some form of
encasement .
</item 3>

<item 4>
Pencil
This method of manufacture , which had been earlier discovered by the Austrian Joseph Hardtmuth , the founder of the Koh - I - Noor in 1790 , remains in use .
In 1802 , the production of graphite leads from graphite and clay was patented by the Koh - I - Noor company in Vienna . In England , pencils continued to be
made from whole sawn graphite . Henry Bessemer 's first successful invention ( 1838 ) was a method of compressing graphite powder into solid graphite thus
allowing the waste from sawing to be reused . The pencil in america Pencil , perhaps made by Henry David Thoreau , in the Concord Museum Pencil manufacturing .
The top sequence shows the old method that required pieces of graphite to be cut to size ; the lower sequence is the new , current method using rods of graphite
and clay . American colonists imported pencils from Europe until after the American Revolution . Benjamin Franklin advertised pencils for sale in his
Pennsylvania Gazette in 1729 , and George Washington used a three - inch pencil when he surveyed the Ohio Territory in 1762 .
</item 4>

<item 5>
Pencil
Their concept involved the hollowing out of a stick of juniper wood . Shortly thereafter , a superior technique was discovered : two wooden halves were carved ,
a graphite stick inserted , and the halves then glued together -- essentially the same method in use to this day . New pencils from graphite powder , and
graphite and clay The first attempt to manufacture graphite sticks from powdered graphite was in Nuremberg , Germany , in 1662 . It used a mixture of graphite ,
sulphur , and antimony . English and German pencils were not available to the French during the Napoleonic Wars ; France , under naval blockade imposed by Great
Britain , was unable to import the pure graphite sticks from the British Grey Knotts mines -- the only known source in the world . France was also unable to
import the inferior German graphite pencil substitute . It took the efforts of an officer in Napoleon 's army to change this . In 1795 , Nicolas - Jacques Conté
discovered a method of mixing powdered graphite with clay and forming the mixture into rods that were then fired in a kiln . By varying the ratio of graphite to
clay , the hardness of the graphite rod could also be varied .
</item 5>

<item 6>
Pencil
Contents ( hide ) 1 History 1.1 Camel 's hair pencil 1.2 Discovery of graphite deposit 1.3 Wood holders added 1.4 New pencils from graphite powder , and
graphite and clay 1.5 The pencil in America 1.6 Eraser attached 1.7 Pencil extenders 2 Types 2.1 By marking material 2.2 By use 2.3 By shape 2.4 By size 2.5 By
manufacture 3 Health 4 Manufacture 5 Grading and classification 5.1 Pencil hardness test 6 External colour and shape of pencils 7 Notable pencil users 8
Manufacturers 9 See also 10 References 11 Bibliography 12 Further reading 13 External links History Old Soviet colored pencils with box ( circa 1959 ) Camel 's
hair pencil Pencil , from Old French pincel , from Latin penicillus a `` little tail '' ( see penis ; pincellus is Latin from the post-classical period )
originally referred to an artist 's fine brush of camel hair , also used for writing before modern lead or chalk pencils .
</item 6>

<item 7>
Pencil
Though the archetypal pencil was an artist 's brush , the stylus , a thin metal stick used for scratching in papyrus or wax tablets , was used extensively by
the Romans and for palm - leaf manuscripts . Discovery of graphite deposit As a technique for drawing , the closest predecesor to the pencil was Silverpoint
until in 1565 ( some sources say as early as 1500 ) , a large deposit of graphite was discovered on the approach to Grey Knotts from the hamlet of Seathwaite in
Borrowdale parish , Cumbria , England . This particular deposit of graphite was extremely pure and solid , and it could easily be sawn into sticks . It remains
the only large - scale deposit of graphite ever found in this solid form . Chemistry was in its infancy and the substance was thought to be a form of lead .
Consequently , it was called plumbago ( Latin for `` lead ore '' ) .
</item 7>

<item 8>
Pencil
Retrieved 23 July 2009 . Jump up ^ `` Pencils , paint and pottery can give you lead poisoning '' . Life . 73 ( 1 ) : 46 . 7 July 1972 . Jump up ^ `` The HB
Graphite Grading Scale '' . Pencils.com . Archived from the original on 21 March 2012 . Retrieved 14 March 2012 . Jump up ^ Petroski , 1990 , `` Appendix A Jump
up ^ `` Staedtler Pencil , video '' . Archived from the original on 21 January 2013 . Retrieved 23 July 2009 . Staedtler.com Jump up ^ Petroski , 1990 Jump up ^
`` Derwent Manufacturing Process '' . Pencils.co.uk . Archived from the original on 17 May 2013 . Retrieved 23 July 2009 . Jump up ^ `` Archived copy '' .
Archived from the original on 6 February 2017 . Retrieved 5 February 2017 .
</item 8>

<item 9>
Pencil
Manufacture The lead of the pencil is a mix of finely ground graphite and clay powders . Before the two substances are mixed , they are separately cleaned of
foreign matter and dried in a manner that creates large square cakes . Once the cakes have fully dried , the graphite and the clay squares are mixed together
using water . The amount of clay content added to the graphite depends on the intended pencil hardness ( lower proportions of clay makes the core softer ) , and
the amount of time spent on grinding the mixture determines the quality of the lead . The mixture is then shaped into long spaghetti - like strings ,
straightened , dried , cut , and then tempered in a kiln . The resulting strings are dipped in oil or molten wax , which seeps into the tiny holes of the
material and allows for the smooth writing ability of the pencil . A juniper or incense - cedar plank with several long parallel grooves is cut to fashion a ``
slat , '' and the graphite / clay strings are inserted into the grooves . Another grooved plank is glued on top , and the whole assembly is then cut into
individual pencils , which are then varnished or painted .
</item 9>

<item 10>
Pencil
p. 419 . Archived from the original on 17 January 2018 . Jump up ^ Steven S. Zumdahl & Susan A. Zumdahl ( 2008 ) . `` No lead pencils '' . Chemistry . Belmont ,
CA : Cengage Learning . p. 343 . ISBN 0 - 547 - 12532 - 1 . Jump up ^ Martin and Jean Norgate , Geography Department , Portsmouth University ( 2008 ) . `` Old
Cumbria Gazetteer , black lead mine , Seathwaite '' . Archived from the original on 7 January 2009 . Retrieved 19 May 2008 . CS1 maint : Multiple names :
authors list ( link ) Jump up ^ Alfred Wainwright ( 2005 ) . A Pictorial Guide to the Lakeland Fells , Western Fells . ISBN 0 - 7112 - 2460 - 9 . Jump up ^ ``
Graphite from the Plumbago Mine , Borrowdale , England '' . Department of Physics at Michigan Technological University .
</item 10>
=== END ITEM id=18c591f2ac71c849 ===

=== ITEM id=5aef6eae6b2cfd3f ===
Gold sentences and facts:
Sentence 0: The film was directed by Robert Mulligan , and starred Gary Grimes as Hermie , Jerry Houser as his best friend Oscy , Oliver Conant as their nerdy young friend Benjie , Jennifer O'Neill as Hermie 's mysterious love interest , and Katherine Allentuck and Christopher Norris as a pair of girls whom Hermie and Oscy attempt to seduce . Mulligan also has an uncredited role as the voice of the adult Hermie . Maureen Stapleton ( Allentuck 's real - life mother ) also appears in a small , uncredited voice role ( calling after Hermie as he leaves the house in an early scene , and after he enters his room in a later scene ) .
  Fact 0.0: Robert Mulligan directed the film.
  Fact 0.1: Gary Grimes starred as Hermie.
  Fact 0.2: Jerry Houser starred as Hermie's best friend Oscy.
  Fact 0.3: Oliver Conant played Benjie, a nerdy young friend of Hermie and Oscy.
  Fact 0.4: Jennifer O'Neill played Hermie's mysterious love interest.
  Fact 0.5: Katherine Allentuck and Christopher Norris played a pair of girls whom Hermie and Oscy attempt to seduce.
  Fact 0.6: Robert Mulligan has an uncredited role as the voice of the adult Hermie.
  Fact 0.7: Maureen Stapleton is Katherine Allentuck's real-life mother.
  Fact 0.8: Maureen Stapleton has a small, uncredited voice role.
  Fact 0.9: Maureen Stapleton's voice calls after Hermie as he leaves the house in an early scene.
  Fact 0.10: Maureen Stapleton's voice calls after Hermie as he enters his room in a later scene.
Sentence 1: Summer of ' 42 Theatrical release poster Directed by Robert Mulligan Produced by Richard A. Roth Written by Herman Raucher Starring Jennifer O'Neill Gary Grimes Jerry Houser Oliver Conant Narrated by Robert Mulligan Music by Michel Legrand Cinematography Robert Surtees Edited by Folmar Blangsted Distributed by Warner Bros. Pictures Release date April 18 , 1971 ( 1971 - 04 - 18 ) Running time 104 minutes Country United States Language English Budget $1 million Box office $32,063,634
  Fact 1.0: Summer of '42 has a theatrical release poster.
  Fact 1.1: Summer of '42 was directed by Robert Mulligan.
  Fact 1.2: Summer of '42 was produced by Richard A. Roth.
  Fact 1.3: Summer of '42 was written by Herman Raucher.
  Fact 1.4: Summer of '42 starred Jennifer O'Neill, Gary Grimes, Jerry Houser and Oliver Conant.
  Fact 1.5: Summer of '42 was narrated by Robert Mulligan.
  Fact 1.6: Summer of '42 had music by Michel Legrand.
  Fact 1.7: Summer of '42 had cinematography by Robert Surtees.
  Fact 1.8: Summer of '42 was edited by Folmar Blangsted.
  Fact 1.9: Summer of '42 was distributed by Warner Bros. Pictures.
  Fact 1.10: Summer of '42 was released on April 18, 1971.
  Fact 1.11: Summer of '42 has a running time of 104 minutes.
  Fact 1.12: Summer of '42 is a United States production.
  Fact 1.13: Summer of '42 is in English.
  Fact 1.14: Summer of '42 had a budget of $1 million.
  Fact 1.15: Summer of '42 had a box office of $32,063,634.

Retrieved items:
<item 1>
Summer of '42
wikipedia.org/w/index.php?title=Summer_of_%2742&oldid=803913855 '' Categories : 1971 films English - language films 1970s comedy - drama films 1970s coming - of
- age films 1970s teen films American biographical films American comedy - drama films American coming - of - age films American films Film scores by Michel
Legrand Films about vacationing Films about virginity Films directed by Robert Mulligan Films set in 1942 Films set in Massachusetts Films set on beaches Films
set on islands Films set on the home front during World War II Films that won the Best Original Score Academy Award Nantucket , Massachusetts Warner Bros.
</item 1>

<item 2>
Summer of '42
Summer of ' 42 - wikipedia Summer of ' 42 Jump to : navigation , search Summer of ' 42 Theatrical release poster Directed by Robert Mulligan Produced by Richard
A. Roth Written by Herman Raucher Starring Jennifer O'Neill Gary Grimes Jerry Houser Oliver Conant Narrated by Robert Mulligan Music by Michel Legrand
Cinematography Robert Surtees Edited by Folmar Blangsted Distributed by Warner Bros. Pictures Release date April 18 , 1971 ( 1971 - 04 - 18 ) Running time 104
minutes Country United States Language English Budget $1 million Box office $32,063,634 Summer of ' 42 is a 1971 American coming - of - age comedy - drama film
based on the memoirs of screenwriter Herman Raucher . It tells the story of how Raucher , in his early teens on his 1942 summer vacation on Nantucket Island (
off the coast of Cape Cod ) , embarks on a one - sided romance with a young woman , Dorothy , whose husband had gone off to fight in World War II .
</item 2>

<item 3>
Summer of '42
The 1973 song `` Summer ( The First Time ) '' by Bobby Goldsboro has almost exactly the same subject and apparent setting , although there is no direct credited
link . Bryan Adams has , however , credited the film as being a partial inspiration for his 1985 hit `` Summer of ' 69 '' . Garth Brooks ' 1993 hit `` That
Summer '' features a similar story of a coming - of - age male in a romance with an older female . Film and television ( edit ) In Stanley Kubrick 's 1980 film
version of Stephen King 's The Shining , Wendy ( Shelley Duvall ) is shown watching Summer of ' 42 on television ( a brief clip of the scene featuring Hermie
helping Dorothy bring her groceries in the house is playing on the television in the background during the scene ) . An episode of the 1970s sitcom Happy Days
was loosely based upon Summer of ' 42 , with Richie Cunningham befriending a Korean War widow . The film 's title formed the basis of a Season 7 episode of The
Simpsons , `` Summer of 4 Ft. 2 '' .
</item 3>

<item 4>
Summer of '42
2008 ( hide ) Films directed by Robert Mulligan Fear Strikes Out ( 1957 ) The Rat Race ( 1960 ) The Great Impostor ( 1961 ) Come September ( 1961 ) The Spiral
Road ( 1962 ) To Kill a Mockingbird ( 1962 ) Love with the Proper Stranger ( 1963 ) Baby the Rain Must Fall ( 1965 ) Inside Daisy Clover ( 1965 ) Up the Down
Staircase ( 1967 ) The Stalking Moon ( 1968 ) The Pursuit of Happiness ( 1971 ) Summer of ' 42 ( 1971 ) The Other ( 1972 ) The Nickel Ride ( 1974 )
Bloodbrothers ( 1978 ) Same Time , Next Year ( 1978 ) Kiss Me Goodbye ( 1982 ) Clara 's Heart ( 1988 ) The Man in the Moon ( 1991 ) Retrieved from ``
https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Summer of '42
Cast ( edit ) Jennifer O'Neill as Dorothy Gary Grimes as Hermie Jerry Houser as Oscy Oliver Conant as Benjie Katherine Allentuck as Aggie Christopher Norris as
Miriam Lou Frizzell as Druggist Robert Mulligan ( uncredited voice ) as Narrator / Older Herman Raucher Walter Scott ( uncredited ) as Dorothy 's husband
Maureen Stapleton ( uncredited voice ) as Hermie 's Mother Production ( edit ) Herman Raucher wrote the film script in the 1950s during his tenure as a
television writer , but `` could n't give it away . '' In the 1960s , he met Robert Mulligan , best known for directing To Kill a Mockingbird . Raucher showed
Mulligan the script , and Mulligan took it to Warner Bros. , where Mulligan argued the film could be shot for the relatively low price of $1 million , and
Warner approved it . They had so little faith in the film becoming a box - office success , though , they shied from paying Raucher outright for the script ,
instead promising him ten percent of the gross . When casting for the role of Dorothy , Warner Bros.
</item 5>

<item 6>
Summer of '42
Alternate sequel ( edit ) In 2002 , O'Neill claimed to have obtained the rights to make a sequel to Summer of ' 42 , based on a short story she wrote , which
took place in an alternate reality where Herman Raucher had a son and divorced his wife , went back to Nantucket in 1962 with a still - living Oscar Seltzer ,
and encountered Dorothy again and married her . As of 2017 , this project -- which O'Neill had hoped to produce with Lifetime television -- has not been
realized , and it is unknown whether O'Neill is still attempting to get it produced , or if Raucher consented to its production . See also ( edit ) List of
American films of 1971 References ( edit ) Jump up ^ `` SUMMER OF ' 42 ( X ) '' . British Board of Film Classification . 1971 - 04 - 15 . Retrieved 2013 - 02 -
06 . Jump up ^ `` Summer of ' 42 , Box Office Information '' . The Numbers . Retrieved January 12 , 2012 . ^ Jump up to : Park , Louis Hillary ( May 2002 ) .
</item 6>

<item 7>
Summer of '42
Though a pop culture phenomenon in the first half of the 1970s , the novelization went out of print and slipped into obscurity throughout the next two decades
until a Broadway adaptation in 2001 brought it back into the public light and prompted Barnes & Noble to acquire the publishing rights to the book . Contents (
hide ) 1 Plot 2 Cast 3 Production 4 Factual basis 5 Reception and awards 6 Sequel 7 Soundtrack 8 Cultural impact 8.1 Music 8.2 Film and television 8.3 Remakes
8.4 Off - Broadway musical 8.5 Alternate sequel 9 See also 10 References 11 External links Plot ( edit ) The film opens with a series of still photographs
appearing over melancholic music , representing the abstract memories of the unseen Herman Raucher , now a middle - aged man . We then hear Raucher recalling
the summer he spent on the island in 1942 . The film flashes back to a day that then 15 - year - old `` Hermie '' and his friends -- jock Oscy and introverted
nerd Benjie -- spent playing on the beach .
</item 7>

<item 8>
Summer of '42
The film is recognized by American Film Institute in these lists : 2002 : AFI 's 100 Years ... 100 Passions -- Nominated 2005 : AFI 's 100 Years of Film Scores
-- Nominated Sequel ( edit ) In 1973 , the film was followed by Class of ' 44 , a slice - of - life film made up of vignettes about Herman Raucher and Oscar
Seltzer 's experiences in college . Class of ' 44 involves the boys facing army service in the last year of World War II . The only crew member from Summer of '
42 to return to the project was Raucher himself , who wrote the script ; a new director and composer were brought in to replace Mulligan and Legrand . Of the
principal four cast members of Summer of ' 42 , only Jerry Houser and Gary Grimes returned for prominent roles , with Oliver Conant making two brief appearances
totaling less than two minutes of screen time . Jennifer O'Neill did not appear in the film at all , nor was the character of Dorothy mentioned . The film is
noted for featuring a young , slim John Candy briefly appearing in his first film role .
</item 8>

<item 9>
Summer of '42
Retrieved September 3 , 2010 . Jump up ^ Internet Movie Database : Summer of ' 42 Business Data IMDb.com Retrieved July 3 , 2006 . Jump up ^ `` Summer of ' 42 (
1971 ) -- Misc Notes '' . TCM.com . Retrieved February 8 , 2012 . Jump up ^ `` AFI 's 100 Years ... 100 Passions Nominees '' ( PDF ) . Retrieved August 20 ,
2016 . Jump up ^ `` AFI 's 100 Years of Film Scores Nominees '' ( PDF ) . Retrieved August 20 , 2016 .
</item 9>

<item 10>
Summer of '42
Archived from the original on January 29 , 2009 . Jump up ^ Jennifer O'Neill in 2002 tv-now.com Retrieved August 11 , 2006 Archived August 29 , 2006 , at the
Wayback Machine . Jump up ^ Theatermania.com , Starry , Starry Morning : Charles Nelson 's Casts and Forecasts . Retrieved September 3 , 2007 Jump up ^ Summer
of ' 42 Plus Twenty usatoday.com . Retrieved July 6 , 2006 . External links ( edit ) www.hermanraucher.com Summer of ' 42 on IMDb Summer of ' 42 at the TCM
Movie Database Summer of ' 42 by Andy Williamson -- The Wordslinger dated April 14 ,
</item 10>
=== END ITEM id=5aef6eae6b2cfd3f ===

=== ITEM id=5ad4537d9acbb967 ===
Gold sentences and facts:
Sentence 0: The United States two - dollar bill ( $2 ) is a current denomination of U.S. currency . The third U.S. President ( 1801 -- 09 ) , Thomas Jefferson , is featured on the obverse of the note . The reverse features an engraving of the painting The Declaration of Independence by John Trumbull . Throughout the $2 bill 's pre-1929 life as a large - sized note , it was issued as a United States Note , National Bank Note , silver certificate , Treasury or `` Coin '' Note and Federal Reserve Bank Note . When U.S. currency was changed to its current size , the $2 bill was issued only as a United States Note . Production went on until 1966 , when the series was discontinued . Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design . Two - dollar bills are seldom seen in circulation as a result of banking policies with businesses which has resulted in low production numbers due to lack of demand . This comparative scarcity in circulation , coupled with a lack of public knowledge that the bill is still in production and circulation , has also inspired urban legends and occasionally has created problems for people trying to use the bill to make purchases .
  Fact 0.0: The United States two-dollar bill ($2) is a current denomination of U.S. currency.
  Fact 0.1: Thomas Jefferson was the third U.S. President.
  Fact 0.2: Thomas Jefferson served as President from 1801 to 1809.
  Fact 0.3: Thomas Jefferson is featured on the obverse of the $2 bill.
  Fact 0.4: The reverse of the $2 bill features an engraving of the painting The Declaration of Independence by John Trumbull.
  Fact 0.5: Before 1929, the $2 bill was a large-sized note.
  Fact 0.6: Before 1929, the $2 bill was issued as a United States Note.
  Fact 0.7: Before 1929, the $2 bill was issued as a National Bank Note.
  Fact 0.8: Before 1929, the $2 bill was issued as a silver certificate.
  Fact 0.9: Before 1929, the $2 bill was issued as a Treasury or "Coin" Note.
  Fact 0.10: Before 1929, the $2 bill was issued as a Federal Reserve Bank Note.
  Fact 0.11: When U.S. currency was changed to its current size, the $2 bill was issued only as a United States Note.
  Fact 0.12: Production of the $2 bill went on until 1966, when the series was discontinued.
  Fact 0.13: Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design.
  Fact 0.14: Two-dollar bills are seldom seen in circulation as a result of banking policies with businesses.
  Fact 0.15: The banking policies with businesses resulted in low production numbers due to lack of demand.
  Fact 0.16: The comparative scarcity of $2 bills in circulation, coupled with a lack of public knowledge that the bill is still in production and circulation, has inspired urban legends.
  Fact 0.17: The comparative scarcity of $2 bills and the lack of public knowledge about them have occasionally created problems for people trying to use the bill to make purchases.

Retrieved items:
<item 1>
United States two-dollar bill
On April 13 , 1976 , the Treasury Department reintroduced the $2 bill as a cost - saving measure . Series 1976 $2 bills were redesigned and issued as a Federal
Reserve Note . The obverse design remains basically unchanged since 1928 and features the same portrait of Jefferson . A green treasury seal and serial numbers
replace the red used on the previous United States Notes . Since the reissue of the bill coincided with the United States Bicentennial , it was decided to use a
bicentennial themed design on the reverse . An engraved rendition ( not an exact reproduction ) of John Trumbull 's The Declaration of Independence replaced
Monticello on the reverse . First day issues of the new $2 bills could be taken to a post office and stamped with the date `` APR 13 1976 '' . In all ,
590,720,000 notes from Series 1976 were printed . Currently , stamped series 1976 $2 bills typically trade for about twice their face value . If the bills were
stamped in an exotic place , the value may be slightly higher .
</item 1>

<item 2>
United States two-dollar bill
When U.S. currency was changed to its current size , the $2 bill was issued only as a United States Note . Production went on until 1966 , when the series was
discontinued . Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design . Two - dollar bills are seldom seen in
circulation as a result of banking policies with businesses which has resulted in low production numbers due to lack of demand . This comparative scarcity in
circulation , coupled with a lack of public knowledge that the bill is still in production and circulation , has also inspired urban legends and occasionally
has created problems for people trying to use the bill to make purchases . Contents ( hide ) 1 Denomination overview 2 Rarity 3 History 3.1 Large - sized notes
3.2 Small size notes 3.2. 1 1928 -- 1966 3.2. 2 1976 -- present 3.3 Series dates 3.3. 1 Large size 3.3.
</item 2>

<item 3>
United States two-dollar bill
Each note was an obligation of the issuing Federal Reserve Bank and could only be redeemed at the corresponding bank . The obverse of the note featured a
borderless portrait of Thomas Jefferson to left and wording in the entire center . The reverse featured a World War I battleship . Beginning in the 1950s ,
production of $2 bills began to decrease . The relative scarcity of the bills lead some to start saving any they found , with the inevitable result that the
notes became rarer in circulation . Currently , the circulation of $2 bills accounts for around 1 % of the U.S. currency in circulation . Small size notes (
edit ) ( 6.14 × 2.61 in ≅ 156 × 66 mm ) 1928 -- 1966 ( edit ) In 1928 , when all U.S. currency was changed to its current size , the $2 bill was issued only as
a United States Note . The obverse featured a cropped version of Thomas Jefferson 's portrait that had been on previous $2 bills . The reverse featured
Jefferson 's home , Monticello . The note 's seal and serial numbers were red .
</item 3>

<item 4>
United States two-dollar bill
In 1976 production of the two - dollar denomination was resumed and the two - dollar bill was finally assigned as a Federal Reserve Note , with a new reverse
design featuring John Trumbull 's depiction of the drafting of the United States Declaration of Independence replacing the previous design of Monticello . It
has remained a current denomination since then . It was estimated that if the two - dollar bill replaced approximately half of the one - dollar bills in
circulation , the federal government would be able to save about $26 million in 1976 dollars ( $112 million adjusted for inflation ) over the period from 1976
to 1981 , due to reduced production , storage , and shipping costs . However , due to their limited use , two - dollar bills are not printed as frequently in a
new series like other denominations , which are produced according to demand . Though some cash registers accommodate the bill , its slot is often used for
things like checks and rolls of coins . Some bill acceptors found in vending machines , self checkout lanes , and other automated kiosks are configured to
accommodate two - dollar bills , even if the fact is not stated on the label .
</item 4>

<item 5>
Large denominations of United States currency
See history of the United States dollar . Passive retirement ( edit ) Although they are still legal tender in the United States , high - denomination bills were
last printed on December 27 , 1945 , and officially discontinued on July 14 , 1969 , by the Federal Reserve System , due to ' lack of use ' . The $5,000 and
$10,000 effectively disappeared well before then . The Federal Reserve began taking high - denomination currency out of circulation ( destroying large bills
received by banks ) in 1969 . As of May 30 , 2009 , only 336 $10,000 bills were known to exist ; 342 remaining $5,000 bills ; and 165,372 remaining $1,000 bills
. Due to their rarity , collectors often pay considerably more than the face value of the bills to acquire them . Some are in museums in other parts of the
world . For the most part , these bills were used by banks and the Federal government for large financial transactions . This was especially true for gold
certificates from 1865 to 1934 .
</item 5>

<item 6>
United States two-dollar bill
The note 's seal and serial numbers were red . The Series of 1928 $2 bill featured the treasury seal superimposed by the United States Note obligation to the
left and a large gray TWO to the right . In 1953 the $2 bill received minor design changes analogous to the $5 United States Note . The treasury seal was made
smaller and moved to the right side of the bill ; it was superimposed over the gray word TWO . The United States Note obligation now became superimposed over a
gray numeral 2 . The reverse remained unchanged . The final change to $2 United States Notes came in 1963 when the motto IN GOD WE TRUST was added to the
reverse over the Monticello . And , because dollar bills were soon to no longer be redeemable in silver , WILL PAY TO THE BEARER ON DEMAND was removed from the
obverse . Two - dollar bills , along with $5 and $100 United States Notes , were officially discontinued in August 1966 , although they remain legal tender .
1976 -- present ( edit ) Series 1976 first day of issue $2 bill with a canceled JFK postage stamp .
</item 6>

<item 7>
United States two-dollar bill
Retrieved February 9 , 2011 . CS1 maint : Uses authors parameter ( link ) Jump up ^ Stone , Suzanne J. ( March -- April 1976 ) . `` The $2 Bill Returns '' .
Retrieved February 9 , 2011 . Jump up ^ Bureau of Engraving and Printing . `` Annual Production Figures '' . Archived from the original on March 4 , 2007 .
Retrieved April 14 , 2007 . Jump up ^ `` Series 2009 $2 '' . USpapermoney.info . Retrieved April 29 , 2012 . Jump up ^ `` 2012 BEP Production Info via FOIA '' .
WheresGeorge.com . Retrieved April 29 , 2012 . Jump up ^ `` Series 2013 $2 '' . USpapermoney.info . Retrieved February 19 , 2014 . Jump up ^ `` 1869 $2 Legal
Tender Rainbow Note '' . usrarecurrency.com . Retrieved November 14 , 2015 . Jump up ^ Kathy Lawrence ( May 19 , 2011 ) .
</item 7>

<item 8>
United States two-dollar bill
Although they are generally available at most banks , two - dollar bills are usually not handed out arbitrarily except upon specific request by the customer ,
and even then , may cause a delay with a trip to the vault . Rarity ( edit ) The seeming rarity of a $2 bill can be attributed to its low printing numbers as a
Federal Reserve Note . Hoarding of the series due to lack of public knowledge of the $2 bill has resulted in very few bills seen in circulation . After its
initial release , supplies of the Series 1976 two - dollar bill were allowed to dwindle until August 1996 when a new series dated 1995 began to be printed .
This series was only printed for the Federal Reserve Bank of Atlanta . Today , there is a common misconception by the general public that the $2 bill is no
longer being produced . According to the Treasury , it `` receives many letters asking why the $2 bill is no longer in circulation '' .
</item 8>

<item 9>
United States two-dollar bill
1 Large size 3.3. 2 Small size 4 Visual snapshot 5 Usage 5.1 Lockheed U-2 5.2 Incidents 6 Uncut currency sheets 7 References 8 External links Denomination
overview ( edit ) The denomination of two dollars was authorized under a congressional act , and first used in March 1862 . The denomination was continuously
used until the 1960s ; by this time the United States Note was the only remaining class of U.S. currency the two - dollar bill was assigned to . In 1966 it was
decided to discontinue all United States Notes from production , which included the two - dollar bill . The two - dollar bill initially was not reassigned to
the Federal Reserve Note class of United States currency and was thus fully discontinued ; the Treasury Department cited the two - dollar bill 's low use and
unpopularity as the reason for not immediately resuming use of the denomination .
</item 9>

<item 10>
United States two-dollar bill
Between 1966 and 1976 , two - dollar notes were not printed . By 1869 , the $2 United States Note was redesigned with the now familiar portrait of Thomas
Jefferson to the left and a vignette of the United States Capitol in the center of the obverse . This note also featured green tinting on the top and left side
of the obverse . Although this note is technically a United States Note , TREASURY NOTE appeared on it instead of UNITED STATES NOTE . The reverse was
completely redesigned . This series was again revised in 1874 , changes on the obverse included removing the green tinting , adding a red floral design around
WASHINGTON D.C. , and changing the term TREASURY NOTE to UNITED STATES NOTE . The 1874 design was also issued as Series of 1875 and 1878 and by 1880 the red
floral design around WASHINGTON D.C. on the United States Note was removed and the serial numbers were changed to blue . This note with the red floral design
was also issued as Series of 1917 but with red serial numbers by that time .
</item 10>
=== END ITEM id=5ad4537d9acbb967 ===
