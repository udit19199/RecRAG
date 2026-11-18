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

=== ITEM id=467353b9602f4bae ===
Gold sentences and facts:
Sentence 0: Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) : 1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957 Women 's ( 9 ) Cross Country ( 9 ) : 1989 , 1990 , 1991 , 1992 , 1993 , 1994 , 1998 , 2009 , 2010 see also : Big East Conference NCAA team championships List of NCAA schools with the most NCAA Division I championships
  Fact 0.0: Men's category: 12 NCAA championships in total.
  Fact 0.1: Men's basketball has 3 championships, in 1985, 2016 and 2018.
  Fact 0.2: Men's cross country has 4 championships, in 1966, 1967, 1968 and 1970.
  Fact 0.3: Men's football has 1 championship, in 2009.
  Fact 0.4: Men's indoor track and field has 3 championships, in 1968, 1971 and 1979.
  Fact 0.5: Men's outdoor track and field has 1 championship, in 1957.
  Fact 0.6: Women's category: 9 NCAA championships in total.
  Fact 0.7: Women's cross country has 9 championships, in 1989, 1990, 1991, 1992, 1993, 1994, 1998, 2009 and 2010.
  Fact 0.8: The text refers the reader to the Big East Conference NCAA team championships list.
  Fact 0.9: The text refers the reader to the list of NCAA schools with the most NCAA Division I championships.
Sentence 1: Basketball ( 3 ) : 1985 , 2016 , 2018
  Fact 1.0: Basketball has 3 championships, in 1985, 2016 and 2018.
Sentence 2: The Villanova Wildcats compete in the Big East Conference and are coached by Jay Wright . The team has traditionally divides its home schedule between its on - campus arena , the William B. Finneran Pavilion , and the Wells Fargo Center in South Philadelphia , for larger draws . During the 2017 - 18 season , the team played its entire home schedule that season at the Wells Fargo Center following the reconstruction of the Pavilion scheduled to be completed in time for the 2018 - 19 school year . Wright has taken the Wildcats to 12 postseason appearances including 7 consecutive trips ( 2005 -- 2011 ) and 5 consecutive trips ( 2013 - 2017 ) to the NCAA Tournament . The 2011 - 12 team struggled and missed postseason but Coach Wright brought a surprising young team back to the 2013 NCAA Tournament . Villanova has appeared in the NIT 17 times , winning in 1994 , and won the Big East Tournament in 1995 , 2015 and 2017 . The Wildcats have appeared in the NCAA Men 's Tournament 37 times , the 8th highest total in NCAA history have reached the Elite Eight 13 times . They have been to the Final Four of the NCAA Division I Men 's Basketball Tournament on five occasions . The 1939 team coached by Al Severance reached the inaugural NCAA Final Four played at the Palestra for the 1939 NCAA Basketball Tournament . All - American Howard Porter led a Wildcat team coached by Jack Kraft to Final Four of the 1971 NCAA University Division Basketball Tournament . Coach Rollie Massimino took Villanova to a surprising 1985 National Championship . Current Coach Jay Wright reached the Final Four at the 2009 NCAA Division I Men 's Basketball Tournament behind Dante Cunningham and Scottie Reynolds . The team most recently won the 2018 NCAA Division I Men 's Basketball Tournament for their third National Championship .
  Fact 2.0: The Villanova Wildcats compete in the Big East Conference.
  Fact 2.1: The Villanova Wildcats are coached by Jay Wright.
  Fact 2.2: The team has traditionally divided its home schedule between its on-campus arena, the William B. Finneran Pavilion, and the Wells Fargo Center in South Philadelphia.
  Fact 2.3: The Wells Fargo Center in South Philadelphia is used for larger draws.
  Fact 2.4: During the 2017-18 season, the team played its entire home schedule at the Wells Fargo Center.
  Fact 2.5: The Finneran Pavilion was being reconstructed during the 2017-18 season.
  Fact 2.6: The Finneran Pavilion reconstruction was scheduled to be completed in time for the 2018-19 school year.
  Fact 2.7: Jay Wright has taken the Wildcats to 12 postseason appearances.
  Fact 2.8: Jay Wright's postseason appearances include 7 consecutive trips to the NCAA Tournament, from 2005 to 2011.
  Fact 2.9: Jay Wright's postseason appearances include 5 consecutive trips to the NCAA Tournament, from 2013 to 2017.
  Fact 2.10: The 2011-12 Villanova team struggled and missed the postseason.
  Fact 2.11: Coach Wright brought a surprising young team back to the 2013 NCAA Tournament.
  Fact 2.12: Villanova has appeared in the NIT 17 times.
  Fact 2.13: Villanova won the NIT in 1994.
  Fact 2.14: Villanova won the Big East Tournament in 1995, 2015 and 2017.
  Fact 2.15: The Wildcats have appeared in the NCAA Men's Tournament 37 times.
  Fact 2.16: The Wildcats' 37 NCAA Men's Tournament appearances are the 8th highest total in NCAA history.
  Fact 2.17: The Wildcats have reached the Elite Eight 13 times.
  Fact 2.18: The Wildcats have been to the Final Four of the NCAA Division I Men's Basketball Tournament on five occasions.
  Fact 2.19: The 1939 Villanova team, coached by Al Severance, reached the inaugural NCAA Final Four.
  Fact 2.20: The inaugural NCAA Final Four was played at the Palestra for the 1939 NCAA Basketball Tournament.
  Fact 2.21: All-American Howard Porter led a Wildcat team, coached by Jack Kraft, to the Final Four of the 1971 NCAA University Division Basketball Tournament.
  Fact 2.22: Coach Rollie Massimino took Villanova to a surprising 1985 National Championship.
  Fact 2.23: Current coach Jay Wright reached the Final Four at the 2009 NCAA Division I Men's Basketball Tournament.
  Fact 2.24: Jay Wright's 2009 Final Four team was led by Dante Cunningham and Scottie Reynolds.
  Fact 2.25: The team most recently won the 2018 NCAA Division I Men's Basketball Tournament.
  Fact 2.26: The 2018 title was the team's third National Championship.
Sentence 3: Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) : 1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957
  Fact 3.0: Men's category: 12 NCAA championships in total.
  Fact 3.1: Men's basketball has 3 championships, in 1985, 2016 and 2018.
  Fact 3.2: Men's cross country has 4 championships, in 1966, 1967, 1968 and 1970.
  Fact 3.3: Men's football has 1 championship, in 2009.
  Fact 3.4: Men's indoor track and field has 3 championships, in 1968, 1971 and 1979.
  Fact 3.5: Men's outdoor track and field has 1 championship, in 1957.

Retrieved items:
<item 1>
Villanova Wildcats
Villanova has won 4 NCAA Men 's Cross Country Team Championships covering 1966 , 1967 , 1968 and 1970 . NCAA Individual Cross Country Champions Victor Zwolak
1963 Patrick Tiernan 2016 Swimming and Diving ( edit ) Main article : Villanova Swimming and Diving In its history , Villanova Swimming and Diving has produced
several Olympians . Most recently Maddy Crippen competed in the 400m individual medley at the 2000 Olympics . At the 2008 Olympics , Villanova was represented
by Kristina Lennox - Silva . Lennox competed for Puerto Rico and swam the 400m freestyle and 200m butterfly . NCAA team Championships ( edit ) Villanova has 21
NCAA team national championships .
</item 1>

<item 2>
Villanova Wildcats
The Wildcat squad remains the lowest overall seed in tournament history to win the championship , and their overall team shooting percentage of 78.6 % remains
an NCAA tournament record for a single game . This Championship Game is often cited among the greatest upsets in college basketball history . Ed Pinckney , who
shot 5 - of - 7 and had 16 points in the game , was named the NCAA Tournament 's Most Outstanding Player . On April 4 , 2016 , the Wildcats won their second
NCAA Championship in Men 's Basketball , the first time in 31 years . They beat the North Carolina Tar Heels 77 - 74 off a buzzer - beater shot by Kris Jenkins
with 0.5 seconds remaining in regulation . Two years later , on April 2 , 2018 , Villanova earned its third national championship beating the Michigan
Wolverines 79 - 62 . Women 's Basketball ( edit ) Main article : Villanova Wildcats women 's basketball 600th win The Villanova Wildcats women 's basketball
team are coached by Harry Perretta , now serving his 34th year as head coach of the team .
</item 2>

<item 3>
Villanova Wildcats
All - American Howard Porter led a Wildcat team coached by Jack Kraft to Final Four of the 1971 NCAA University Division Basketball Tournament . Coach Rollie
Massimino took Villanova to a surprising 1985 National Championship . Current Coach Jay Wright reached the Final Four at the 2009 NCAA Division I Men 's
Basketball Tournament behind Dante Cunningham and Scottie Reynolds . The team most recently won the 2018 NCAA Division I Men 's Basketball Tournament for their
third National Championship . Villanova won the 1985 NCAA Division I Men 's Basketball Tournament with a historic 66 - 64 win over top - seeded Georgetown . The
eighth - seeded Wildcats ( unranked in the final AP poll ) beat Dayton ( at Dayton ) , top - seeded Michigan , Maryland and second - seeded North Carolina to
win the Southeast Regional en route to the Final Four in Lexington , Kentucky . After defeating 2 - seed Memphis State ( now known as Memphis ) in the national
semifinals , Villanova met defending champion and ten - point - favorite Georgetown , led by Patrick Ewing , in the title game on April 1 , 1985 .
</item 3>

<item 4>
Villanova Wildcats
Villanova was the Big East regular season and tournament champions in both 1985 - 86 and 1986 - 87 behind All - American Shelly Pennefather . During the 2008 -
09 season , Coach Perretta led the Wildcats to the NCAA Tournament for the ninth time in his Villanova career . This marked the ninth postseason appearance in
the previous 10 years for the Wildcats . After playing in the 2000 WNIT , Villanova went to four consecutive NCAA Tournaments from 2001 to 2004 before playing
in the WNIT in 2005 , 2006 and 2008 . The 2002 - 03 season was a special one for the Wildcats . In addition to an impressive 28 - 6 overall record and a 12 - 4
Big East Conference mark , the Wildcats won their third Big East Conference Tournament championship and first since 1987 with an improbable 52 - 48 upset over
previously unbeaten Connecticut . The win snapped Connecticut 's then NCAA Division I women 's basketball record of 70 - straight victories and ended
Connecticut 's streak of nine - straight conference tournament championships .
</item 4>

<item 5>
Villanova Wildcats
The women 's team has also had a multitude of success , producing 10 Big East team Championships and 7 NCAA team Championships . They have also produced 7
Olympians including Vicki Huber , Sonia O'Sullivan , Kim Certain , Kate Fonshell , Jen Rhines , Carmen Douma , and Carrie Tollefson . The Women 's team has won
28 Penn Relays Championships , which is the most wins by any women 's program . The current women 's coaches include head coach , Gina Procaccio , and assistant
head coach , Anthony Williams . At least one Villanova athlete has competed in every Summer Olympics since 1948 , winning a total of 13 medals ( 9 gold , 4
silver ) . Cross Country ( edit ) Villanova Men 's and Women 's Cross Country teams have a successful history of Big East and NCAA National Championships .
Villanova won both the Men 's and Women 's Big East Cross Country Team Titles in 2011 . The 2011 women 's Cross Country team finished as national runner - up
led by Sheila Reid individual title .
</item 5>

<item 6>
Villanova Wildcats
For the 2017 -- 18 season , they will play their home games at Jake Nevin Field House on the Villanova campus while the future Finneran Pavilion is undergoing
renovations . The Women 's basketball program started in 1969 and played under CIAW and AIAW through 1981 - 82 season before moving to NCAA Division I. In the
81 - 82 season , the Wildcats posted an outstanding 29 - 4 record and reached the Final Four of the 1982 AIAW Women 's Basketball Tournament before losing to
eventual National Champion , Rutgers . Through the 2009 - 10 season , the Villanova Women have a 669 - 413 alltime record since the program started in 1969 .
Previous head coaches include Liz Cawley ( 1969 -- 70 ) , Jane Sefranek - Stoltz ( 1970 -- 75 ) and Joan King ( 1975 -- 78 ) . The 1982 - 83 season was the
initial season for Villanova Women 's basketball to play NCAA Division I and they joined the Big East Conference that same season .
</item 6>

<item 7>
Villanova Wildcats
Track and field ( edit ) Villanova University 's Track and field team has a history of athletic success that has spanned from championships in both incarnations
of the Big East Conference to NCAA championships . The men 's team has produced 69 NCAA championships , 36 indoor and 33 outdoor . The team has had 8 NCAA team
championships -- 4 cross country , 3 indoor ( ' 68 , ' 71 , ' 79 ) , 1 outdoor ( ' 57 ) . Prior to NCAA sponsoring Indoor T&F in 1965 , Villanova won 6 ICAAAA
Indoor T&F Championships ( ' 57 , ' 58 , ' 60 , ' 62 , ' 63 , ' 64 ) . Villanova has produced 28 athletes who have made appearances in the Olympics , 10 of whom
have medaled ( 7 Gold medals , 3 Silver medals ) . The men 's team has also won 112 Penn Relays Championships , which stands as the most wins by any school .
The men 's current coaches include head coach , Marcus O'Sullivan , and assistant head coach , Anthony Williams .
</item 7>

<item 8>
Villanova Wildcats
NCAA team Championships ( edit ) Villanova has 21 NCAA team national championships . Men 's ( 12 ) Basketball ( 3 ) : 1985 , 2016 , 2018 Cross Country ( 4 ) :
1966 , 1967 , 1968 , 1970 Football ( 1 ) : 2009 Indoor Track & Field ( 3 ) : 1968 , 1971 , 1979 Outdoor Track & Field ( 1 ) : 1957 Women 's ( 9 ) Cross Country
( 9 ) : 1989 , 1990 , 1991 , 1992 , 1993 , 1994 , 1998 , 2009 , 2010 see also : Big East Conference NCAA team championships List of NCAA schools with the most
NCAA Division I championships Conference realignment ( edit ) In December 2012 the presidents of seven Big East schools which do not sponsor FBS football ( also
sometimes referred to as the `` Catholic 7 '' ) , including Villanova , voted unanimously to leave the Big East and form a new conference .
</item 8>

<item 9>
Villanova Wildcats
During the 2017 - 18 season , the team played its entire home schedule that season at the Wells Fargo Center following the reconstruction of the Pavilion
scheduled to be completed in time for the 2018 - 19 school year . Wright has taken the Wildcats to 12 postseason appearances including 7 consecutive trips (
2005 -- 2011 ) and 5 consecutive trips ( 2013 - 2017 ) to the NCAA Tournament . The 2011 - 12 team struggled and missed postseason but Coach Wright brought a
surprising young team back to the 2013 NCAA Tournament . Villanova has appeared in the NIT 17 times , winning in 1994 , and won the Big East Tournament in 1995
, 2015 and 2017 . The Wildcats have appeared in the NCAA Men 's Tournament 37 times , the 8th highest total in NCAA history have reached the Elite Eight 13
times . They have been to the Final Four of the NCAA Division I Men 's Basketball Tournament on five occasions . The 1939 team coached by Al Severance reached
the inaugural NCAA Final Four played at the Palestra for the 1939 NCAA Basketball Tournament .
</item 9>

<item 10>
Villanova Wildcats
Coach Michael Corrado has taken the team to its only two NCAA appearances in history during his 6 - year tenure and the school is poised for greater success in
upcoming seasons . Villanova rose as high as # 7 ranking and qualified for an at - large bid in the 2011 NCAA Tournament where they lost to the University of
Denver in the first round . During the regular season , the Wildcats drew over 6500 fans to a Big East match - up with Syracuse setting a new on - campus record
for Lacrosse . Brian Karalunas became the first USILA First Team All - American in Villanova history . Through 2009 , Villanova men 's lacrosse was a member of
the Colonial Athletic Association and won the CAA tournament for their first title in 2009 . The fourth - seeded Wildcats were the lowest - seeded championship
team in conference history . That season , Villanova also made their first NCAA tournament appearance , where they were routed in the first round by first -
seeded Virginia , 18 -- 6 . The 2004 team finished 13 - 3 playing against the stiff competition of the CAA .
</item 10>
=== END ITEM id=467353b9602f4bae ===

=== ITEM id=58f20d2d1f559192 ===
Gold sentences and facts:
Sentence 0: The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ; traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising , when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
  Fact 0.0: The Sino-Indian War is also known as the Sino-Indian Border Conflict.
  Fact 0.1: The Sino-Indian War's Hindi name is भारत - चीन युद्ध (Bhārat - Chīn Yuddh).
  Fact 0.2: The Sino-Indian War's simplified Chinese name is 中印边境战争.
  Fact 0.3: The Sino-Indian War's traditional Chinese name is 中印邊境戰爭.
  Fact 0.4: The Sino-Indian War's pinyin name is Zhōng-Yìn Biānjìng Zhànzhēng.
  Fact 0.5: The Sino-Indian War was a war between China and India.
  Fact 0.6: The Sino-Indian War occurred in 1962.
  Fact 0.7: A disputed Himalayan border was the main pretext for the Sino-Indian War.
  Fact 0.8: Other issues played a role in the Sino-Indian War.
  Fact 0.9: There had been a series of violent border incidents after the 1959 Tibetan uprising.
  Fact 0.10: India had granted asylum to the Dalai Lama at the time of the 1959 Tibetan uprising.
  Fact 0.11: India initiated a Forward Policy.
  Fact 0.12: In its Forward Policy, India placed outposts along the border.
  Fact 0.13: In its Forward Policy, India placed several outposts north of the McMahon Line.
  Fact 0.14: The McMahon Line is the eastern portion of a Line of Actual Control.
  Fact 0.15: The Line of Actual Control was proclaimed by Chinese Premier Zhou Enlai in 1959.

Retrieved items:
<item 1>
Sino-Indian War
Unable to reach political accommodation on disputed territory along the 3,225 - kilometre - long Himalayan border , the Chinese launched simultaneous offensives
in Ladakh and across the McMahon Line on 20 October 1962 , coinciding with the Cuban Missile Crisis . Chinese troops advanced over Indian forces in both
theatres , capturing Rezang la in Chushul in the western theatre , as well as Tawang in the eastern theatre . The war ended when China declared a ceasefire on
20 November 1962 , and simultaneously announced its withdrawal to its claimed ' line of actual control ' . The Sino - Indian War is notable for the harsh
mountain conditions under which much of the battle took place , entailing large - scale combat at altitudes of over 4,000 metres ( 14,000 feet ) . The Sino -
Indian War was also noted for the non-deployment of the navy or air force by either the Chinese or Indian side .
</item 1>

<item 2>
Graph query row:
war: Sino-Indian War
war_date: 20 October -- 21 November 1962
war_score: None
relationship: RELATED_TO
connected_entity: China
outcome: None
result: None
</item 2>

<item 3>
Sino-Indian War
The Sino - Indian War ( Hindi : भारत - चीन युद्ध Bhārat - Chīn Yuddh ) , also known as the Sino - Indian Border Conflict ( simplified Chinese : 中 印 边境 战争 ;
traditional Chinese : 中 印 邊境 戰爭 ; pinyin : Zhōng - Yìn Biānjìng Zhànzhēng ) , was a war between China and India that occurred in 1962 . A disputed Himalayan
border was the main pretext for war , but other issues played a role . There had been a series of violent border incidents after the 1959 Tibetan uprising ,
when India had granted asylum to the Dalai Lama . India initiated a Forward Policy in which it placed outposts along the border , including several north of the
McMahon Line , the eastern portion of a Line of Actual Control proclaimed by Chinese Premier Zhou Enlai in 1959 .
</item 3>

<item 4>
Sino-Indian War
According to Patterson , the Indian motives were threefold : Test Chinese resolve and intentions regarding India . Test whether India would enjoy Soviet backing
in the event of a Sino - Indian war . Create sympathy for India within the U.S. , with whom relations had deteriorated after the Indian annexation of Goa . On
10 July 1962 , 350 Chinese troops surrounded an Indian post in Chushul ( north of the McMahon Line ) but withdrew after a heated argument via loudspeaker . On
22 July , the Forward Policy was extended to allow Indian troops to push back Chinese troops already established in disputed territory . Whereas Indian troops
were previously ordered to fire only in self - defence , all post commanders were now given discretion to open fire upon Chinese forces if threatened . In
August , the Chinese military improved its combat readiness along the McMahon Line and began stockpiling ammunition , weapons and gasoline . Given his
foreknowledge of the coming Cuban Missile Crisis , Mao Zedong was able to persuade Nikita Khrushchev to reverse the Russian policy of backing India , at least
temporarily . In mid-October , the Communist organ Pravda encouraged peace between India and China .
</item 4>

<item 5>
Sino-Indian War
It is noteworthy that the buildup and offensive from China occurred concurrently with the 13 - day Cuban Missile Crisis ( 16 -- 28 October 1962 ) that saw both
the United States and the Soviet Union confronting each other , and India did not receive assistance from either of these world powers until the Cuban Missile
Crisis was resolved . Contents ( hide ) 1 Location 2 Background 2.1 Aksai Chin 2.2 The McMahon Line 3 Events leading up to war 3.1 Tibet and the border dispute
3.2 1960 meetings to resolve the boundary question 3.3 The Forward Policy 3.4 Early incidents 4 Confrontation at Thag La 5 Chinese and Indian preparations 5.1
Motives 5.2 Military planning 6 Chinese offensive 6.1 Eastern theatre 6.2 Western theatre 7 Lull in the fighting 8 Continuation of war 8.1 Eastern theatre 8.2
Western theatre 9 Ceasefire 10 World opinion 11 Involvement of other nations 12 Aftermath 12.1 China 12.2 India 12.2.
</item 5>

<item 6>
Graph query row:
war: Sino-Indian War
war_date: 20 October -- 21 November 1962
war_score: None
relationship: LOCATED_IN
connected_entity: Himalayan border
outcome: None
result: None
</item 6>

<item 7>
Sino-Indian War
Retrieved 4 November 2010 . Bibliography ( edit ) Das Gupta , Amit R. ; Lüthi , Lorenz M. , eds. ( 2016 ) , The Sino - Indian War of 1962 : New perspectives ,
Taylor & Francis , ISBN 978 - 1 - 315 - 38892 - 2 Lüthi , Lorenz M. ( 2016 ) , `` India 's Relations with China , 1945 -- 1974 '' , Ibid , pp. 29 -- 47 Chaowu ,
Dai ( 2016 ) , `` From ' Hindi - Chini Bhai - Bhai ' to ' international class struggle ' against Nehru : China 's India policy and the frontier dispute , 1950
-- 62 '' , Ibid , pp. 68 -- 84 Fisher , Margaret W. ; Rose , Leo E. ; Huttenback , Robert A.
</item 7>

<item 8>
Sino-Indian War
Sino - Indian war - Wikipedia Sino - Indian war Jump to : navigation , search The examples and perspective in this article may not include all significant
viewpoints . Please improve the article or discuss the issue . ( September 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War The
Sino - Indian War was fought between India and China Date 20 October -- 21 November 1962 ( 1 month and 1 day ) Location Aksai Chin and North - East Frontier
Agency Result Chinese victory Belligerents India China Commanders and leaders Brij Mohan Kaul ( Chief of General Staff of the Indian Army ) Sarvepalli
Radhakrishnan ( President of India ) Jawaharlal Nehru ( Prime Minister of India ) V.K.
</item 8>

<item 9>
Sino-Indian War
( March 2017 ) ( Learn how and when to remove this template message ) Sino - Indian War ( 1962 ) Remembering a War : The 1962 India - China Conflict --
Rediff.com . Neville Maxwell : Henderson Brooks Report 1962 Sino - Indian War , Hindustan Times Why India lost the 1962 border war ? -- Tejas Patel War in the
Himalayas : 1962 Indo - Sino Conflict ( includes official war history ) from History Division , Ministry of Defence , Government of India Critical Asian Studies
Article : Sino Indian War 1962 India , China to speed up border dispute talks : 2005 Xinhuanet The Rediff Special / Claude Arpi 1962 War and Its Implications fr
Sino - India Relations Historical maps of the Sino - Indian border ( in Chinese ) Conflict in Kashmir : Selected Internet Resources by the Library , University
of California , Berkeley , USA ; ( ( University of California , Berkeley Library Bibliographies and Web - Bibliographies list ) Frontier India India - China
Section China , India , and the fruits of Nehru 's folly by Venkatesan Vembu , Daily News & Analysis , 6 June
</item 9>

<item 10>
Sino-Indian War
China 's heavy artillery facing Taiwan could then be moved to Tibet . It took China six to eight months to gather the resources needed for the war , according
to Anil Athale , author of the official Indian history . The Chinese sent a large quantity of non-military supplies to Tibet through the Indian port of Calcutta
. Early incidents ( edit ) Various border conflicts and `` military incidents '' between India and China flared up throughout the summer and autumn of 1962 . In
May , the Indian Air Force was told not to plan for close air support , although it was assessed as being a feasible way to counter the unfavourable ratio of
Chinese to Indian troops . In June , a skirmish caused the deaths of dozens of Chinese troops . The Indian Intelligence Bureau received information about a
Chinese buildup along the border which could be a precursor to war . During June -- July 1962 , Indian military planners began advocating `` probing actions ''
against the Chinese , and accordingly , moved mountain troops forward to cut off Chinese supply lines . According to Patterson , the Indian motives were
threefold : Test Chinese resolve and intentions regarding India .
</item 10>
=== END ITEM id=58f20d2d1f559192 ===

=== ITEM id=485e1811240188f2 ===
Gold sentences and facts:
Sentence 0: Nintendo claims that the design of the Pro Controller is an `` enhanced version '' of the Classic Controller and `` offers a richer experience '' . Certain games with flexible control schemes , such as Call of Duty : Black Ops II and Trine 2 : Director 's Cut , are also compatible with the Classic Controller . The Wii U Pro Controller is not compatible with previous - generation Wii games .
  Fact 0.0: Nintendo claims that the design of the Wii U Pro Controller is an enhanced version of the Classic Controller.
  Fact 0.1: Nintendo claims that the Wii U Pro Controller offers a richer experience.
  Fact 0.2: Certain games with flexible control schemes are compatible with the Classic Controller.
  Fact 0.3: Call of Duty: Black Ops II is a game with a flexible control scheme that is compatible with the Classic Controller.
  Fact 0.4: Trine 2: Director's Cut is a game with a flexible control scheme that is compatible with the Classic Controller.
  Fact 0.5: The Wii U Pro Controller is not compatible with previous-generation Wii games.
Sentence 1: The Wii U Pro Controller ( Wii U PRO コントローラー ) is a video game controller produced by Nintendo for the Wii U video game console .
  Fact 1.0: The Wii U Pro Controller is a video game controller.
  Fact 1.1: The Wii U Pro Controller is produced by Nintendo.
  Fact 1.2: The Wii U Pro Controller is made for the Wii U video game console.
Sentence 2: The controller functions as a secondary controller released for the console , available separately . The Wii U system can be connected to up to four Wii U Pro Controllers at one time , however Super Smash Bros. for Wii U allows the usage of more than 4 . Like more traditional controllers , it features standard analog sticks and face buttons . Like the Wii U GamePad and Classic Controller Pro and unlike other eighth generation controllers ( i.e. , the DualShock 4 and the Xbox One Controller ) , the triggers are digital ( i.e. , not analog ) .
  Fact 2.0: The Wii U Pro Controller functions as a secondary controller for the Wii U.
  Fact 2.1: The Wii U Pro Controller was released for the console.
  Fact 2.2: The Wii U Pro Controller is available separately.
  Fact 2.3: The Wii U console can be connected to up to four Wii U Pro Controllers at one time.
  Fact 2.4: Super Smash Bros. for Wii U allows the use of more than four controllers.
  Fact 2.5: The Wii U Pro Controller has standard analog sticks.
  Fact 2.6: The Wii U Pro Controller has face buttons, like more traditional controllers.
  Fact 2.7: The Wii U Pro Controller has digital triggers, not analog triggers.
  Fact 2.8: The Wii U GamePad has digital triggers.
  Fact 2.9: The Classic Controller Pro has digital triggers.
  Fact 2.10: Unlike the DualShock 4 and the Xbox One Controller, the Wii U Pro Controller has digital triggers.

Retrieved items:
<item 1>
Wii U Pro Controller
Tomshardware.com . Retrieved November 26 , 2012 . Wii U Hardware Wii U GamePad Off - TV Play Wii U Pro Controller Wii accessories Wii Remote ( Plus ) MotionPlus
Nunchuk Classic Controller Balance Board Wii Speak Third - party accessories Amiibo Online service Nintendo Network Miiverse Nintendo eShop Nintendo TVii Amazon
Instant Video / LoveFilm Hulu Plus Netflix TiVo SpotPass System software Home Menu Amazon Instant Video LoveFilm Crunchyroll Hulu Napster Netflix Niconico Mii
Maker YouTube Wii Mode Wii U Chat Games List Best - selling Balance Board Pro Controller Miis Nintendo Network Wii MotionPlus Virtual Console Japan North
America PAL region Backward compatibility Wii games Wi - Fi Connection Wii applications WiiWare games North America PAL region Wii Mode Virtual Console Japan
North America PAL region Wii series Street U Fit U Party U Sports Club Miscellaneous List of Wii games on Wii U eShop Licensed toolkits Integrated Nintendo Web
Framework Unity Supported Autodesk Gameware CryEngine ( 4th generation ) Havok Gamebryo Umbra Unreal Engine 3 ← Predecessor : Wii Follow - up : Nintendo Switch
→ Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
History of Nintendo
During the Electronics Entertainment Expo in 2011 , Nintendo 's press conference put a great amount of stress on the abilities and innovations of the new
controller , a unit resembling a tablet . Many gamers were confused after the press conference about whether the Wii U is an entirely new system or just a new
controller for the Wii . The Wii U was , however , a whole new home console system developed by Nintendo . The system was released on November 18 , 2012 in
North America , November 30 , 2012 in Europe and Australia , and on December 8 , 2012 in Japan . It featured a new controller with a 6.2 inch touchscreen , two
thumb pads , four shoulder buttons , a directional pad , and four lettered buttons . It also featured an accelerometer , gyroscope , camera , Near - field
communications ( NFC ) , and built - in microphone , similar those in the Nintendo 3DS . The Nintendo Switch Console in the Dock with Red and Blue Joy - Con
controllers . On July 11 , 2015 , Satoru Iwata died from a bile duct tumor at the age of 55 .
</item 2>

<item 3>
Wii U Pro Controller
nintendolife.com . October 3 , 2013 . Retrieved August 16 , 2014 . Jump up ^ `` Wii U Pro Controller Boasts 80 - Hour Battery Life - IGN '' . Uk.ign.com .
October 16 , 2012 . Retrieved September 21 , 2013 . Jump up ^ Liebl , Matt ( June 12 , 2012 ) . `` Nintendo defends Wii U Pro Controller design '' . GameZone .
Retrieved July 28 , 2012 . Jump up ^ `` Wii U Pro Controller Not Compatible With Wii Games '' . Tomshardware.com . Retrieved November 26 , 2012 .
</item 3>

<item 4>
Wii U Pro Controller
The Wii U Pro Controller is not compatible with previous - generation Wii games . References ( edit ) Jump up ^ Franich , Darren ( June 3 , 2012 ) . `` The Wii
U Pro Controller looks very familiar ... '' EW.com . Retrieved July 28 , 2012 . Jump up ^ Kain , Erik ( June 5 , 2012 ) . `` Nintendo 's Smart Decision With Its
Wii U Pro Controller '' . Forbes . Retrieved July 28 , 2012 . Jump up ^ Grubb , Jeffrey ( June 3 , 2012 ) . `` Nintendo 's Wii U Pro Controller may make
Microsoft blush '' . VentureBeat . Retrieved July 28 , 2012 . Jump up ^ How to Sync a Wii Remote with a Wii U Console Jump up ^ `` Wii U Pro Controller and 3DS
share the same battery '' . gonintendo.com . November 9 , 2012 . Retrieved August 16 , 2014 . Jump up ^ `` We Opened Up The Nintendo 2DS And Found A 1300mAh
Battery Inside '' .
</item 4>

<item 5>
Wii U Pro Controller
Features ( edit ) See also : List of Wii U games that use the Wii U Pro Controller The controller functions as a secondary controller released for the console ,
available separately . The Wii U system can be connected to up to four Wii U Pro Controllers at one time , however Super Smash Bros. for Wii U allows the usage
of more than 4 . Like more traditional controllers , it features standard analog sticks and face buttons . Like the Wii U GamePad and Classic Controller Pro and
unlike other eighth generation controllers ( i.e. , the DualShock 4 and the Xbox One Controller ) , the triggers are digital ( i.e. , not analog ) . The Wii U
Pro Controller uses the same 1300mAh CTR - 003 battery found in the Nintendo 3DS and 2DS , which can last up to 80 hours before needing to be recharged .
Nintendo claims that the design of the Pro Controller is an `` enhanced version '' of the Classic Controller and `` offers a richer experience '' . Certain
games with flexible control schemes , such as Call of Duty : Black Ops II and Trine 2 : Director 's Cut , are also compatible with the Classic Controller .
</item 5>

<item 6>
Wii U Pro Controller
Wii U Pro Controller - wikipedia Wii U Pro Controller Jump to : navigation , search Wii U Pro Controller Wii U Pro Controller Manufacturer Nintendo Type Video
game controller Generation Eighth generation Retail availability November 18 , 2012 Input 2 × analog sticks Digital d - pad 10 × digital face buttons ( two
buttons found under analog sticks ; controller features additional button for syncing purposes ) 2 × digital shoulder buttons 2 × digital triggers Connectivity
Bluetooth Power 1300 mAh , Mini USB B connector ( recharge ) Predecessor Wii Remote ( Plus ) and Classic Controller ( Pro ) Successor Nintendo Switch Pro
Controller The Wii U Pro Controller ( Wii U PRO コントローラー ) is a video game controller produced by Nintendo for the Wii U video game console . History ( edit )
Nintendo unveiled the Wii U Pro Controller at E3 2012 ; the controller is aimed towards attracting `` hardcore '' players to the platform with a more
traditional design , to make the Wii U more competitive with Sony 's and Microsoft 's offerings . Many video game journalists have noted the similarity between
the controller and Microsoft 's Xbox 360 Controller .
</item 6>

<item 7>
Nintendo’s Wii U Pro Controller, released separately on November 18, 2012, was designed to appeal to players who preferred a traditional controller. It
resembles the Xbox 360 controller and has two analog sticks, a digital D-pad, face buttons, digital shoulder buttons and triggers, and Bluetooth connectivity. A
1300mAh rechargeable battery provides up to 80 hours of play. A Wii U can connect up to four controllers, though *Super Smash Bros. for Wii U* supports more
</item 7>

<item 8>
The Wii U Pro Controller does not work with previous-generation Wii games. Contemporary coverage discussed its familiar design, Nintendo’s defense of that
design, and reports of an approximately 80-hour battery life; one source said it shared a battery with the 3DS. References also covered syncing Wii Remotes with
a Wii U.
</item 8>

<item 9>
Wii U Pro Controller
wikipedia.org/w/index.php?title=Wii_U_Pro_Controller&oldid=837880962 '' Categories : Nintendo controllers Wii U Hidden categories : Pages using deprecated image
syntax Articles containing Japanese - language text Talk About Wikipedia Deutsch Español Italiano Português Edit links This page was last edited on 23 April
2018 , at 16 : 07 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
The excerpt is a 2018 Wikipedia page footer and navigation list for the Wii U Pro Controller. It links to Wii U hardware, accessories, online services,
software, games, backward compatibility, and supported development tools and engines. It identifies Wii as the predecessor and Nintendo Switch as the follow-up.
The page was last edited April 23, 2018, and is available under a Creative Commons Attribution-ShareAlike license.
</item 10>
=== END ITEM id=485e1811240188f2 ===

=== ITEM id=9fb8a96ac4250804 ===
Gold sentences and facts:
Sentence 0: Life of Pi is a Canadian fantasy adventure novel by Yann Martel published in 2001 . The protagonist is Piscine Molitor `` Pi '' Patel , an Indian boy from Pondicherry who explores issues of spirituality and practicality from an early age . He survives 227 days after a shipwreck while stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker .
  Fact 0.0: Life of Pi is a Canadian fantasy adventure novel.
  Fact 0.1: Life of Pi is a novel by Yann Martel.
  Fact 0.2: Life of Pi was published in 2001.
  Fact 0.3: The protagonist of Life of Pi is Piscine Molitor 'Pi' Patel.
  Fact 0.4: Piscine Molitor 'Pi' Patel is an Indian boy from Pondicherry.
  Fact 0.5: Piscine Molitor 'Pi' Patel explores issues of spirituality and practicality from an early age.
  Fact 0.6: Pi survives 227 days after a shipwreck.
  Fact 0.7: Pi is stranded on a lifeboat in the Pacific Ocean.
  Fact 0.8: Pi is stranded on the lifeboat with a Bengal tiger named Richard Parker.

Retrieved items:
<item 1>
Life of Pi
The cook amputates the sailor 's leg for use as fishing bait , then kills the sailor himself as well as Pi 's mother for food , and soon he is killed by Pi ,
who dines on him . The investigators note parallels between the two stories . They soon conclude that the hyena symbolizes the cook , the zebra the sailor , the
orangutan Pi 's mother , and the tiger represents Pi . Pi points out that neither story can be proven and neither explains the cause of the shipwreck , so he
asks the officials which story they prefer : the one without animals or the one with animals . They eventually choose the story with the animals . Pi thanks
them and says : `` And so it goes with God . '' The investigators then leave and file a report . Major themes ( edit ) Life is a story ( edit ) Life of Pi ,
according to Yann Martel , can be summarized in three statements : `` Life is a story ... You can choose your story ... A story with God is the better story .
'' A recurring theme throughout the novel seems to be believability .
</item 1>

<item 2>
Life of Pi
Some time later , Pi 's boat comes ashore on a floating island network of algae and inhabited by hundreds of thousands of meerkats . Soon , Pi and Richard
Parker regain strength , but the boy 's discovery of the carnivorous nature of the island 's plant life forces him to return to the ocean . Two hundred and
twenty - seven days after the ship 's sinking , the lifeboat washes onto a beach in Mexico , after which Richard Parker disappears into the nearby jungle
without looking back , leaving Pi heartbroken at the abrupt farewell . Part three ( edit ) The third part of the novel describes a conversation between Pi and
two officials from the Japanese Ministry of Transport , who are conducting an inquiry into the shipwreck . They meet him at the hospital in Mexico where he is
recovering . Pi tells them his tale , but the officials reject it as unbelievable . Pi then offers them a second story in which he is adrift on a lifeboat not
with zoo animals , but with the ship 's cook , a Taiwanese sailor with a broken leg , and his own mother .
</item 2>

<item 3>
Life of Pi
He was named after a swimming pool in Paris , despite the fact that neither his mother nor his father particularly liked swimming . The story is told as a
narrative from the perspective of a middle - aged Pi , now married with his own family , and living in Canada . At the time of main events of the story , he is
sixteen years old . He recounts the story of his life and his 227 - day journey on a lifeboat when his ship sinks in the middle of the Pacific Ocean during a
voyage to North America . Richard Parker ( edit ) A Bengal tiger Richard Parker is a royal Bengal tiger who is stranded on the lifeboat with Pi when the ship
sinks . Richard Parker lives on the lifeboat with Pi and is kept alive with the food and water Pi delivers . Richard Parker develops a relationship with Pi that
allows them to coexist in their struggle . In the novel , a hunter named Richard Parker is hired to kill a panther thought to have killed seven people within
two months . Instead , he accidentally immobilizes a female Bengal tiger with tranquilizer darts while her cub escapes hiding in a bush . Parker names the cub
Thirsty after his enthusiasm when drinking from a nearby river .
</item 3>

<item 4>
Life of Pi
Suddenly emerging from his hideaway , Richard Parker kills and eats the hyena . Frightened , Pi constructs a small raft out of rescue flotation devices ,
tethers it to the bow of the boat and makes it his place of retirement . He begins conditioning Richard Parker to take a submissive role by using food as a
positive reinforcer , and seasickness as a punishment mechanism , while using a whistle for signals . Soon , Pi asserts himself as the alpha animal , and is
eventually able to share the boat with his feline companion , admitting in the end that Richard Parker is the one who helped him survive his ordeal . Pi
recounts various events while adrift in the Pacific Ocean . At his lowest point , exposure renders him blind and unable to catch fish . In a state of delirium ,
he talks with a marine `` echo '' , which he initially identifies as Richard Parker having gained the ability to speak , but it turns out to be another blind
castaway , a Frenchman , who boards the lifeboat with the intention of killing and eating Pi , but is eventually killed by Richard Parker .
</item 4>

<item 5>
Yann Martel’s Canadian novel *Life of Pi*, published in 2001, follows Piscine “Pi” Patel, an Indian boy whose ship sinks while his family emigrates. He survives
227 days adrift in the Pacific in a lifeboat with Bengal tiger Richard Parker. Blending adventure, spirituality and questions about storytelling and belief, it
uses alternating perspectives and an ambiguous ending. The bestseller won the 2002 Man Booker Prize and other awards. Adaptations include an
</item 5>

<item 6>
Yann Martel’s novel follows Piscine “Pi” Patel, raised in Pondicherry amid Hinduism, Christianity, Islam, and his family’s zoo. After emigrating to Canada, Pi
survives the sinking of a freighter, drifting 227 days with Bengal tiger Richard Parker. Their struggle tests his ingenuity, faith, and resilience. Rescued in
Mexico, Pi offers investigators two accounts: one involving animals and a darker version involving human survivors. They choose the animal story
</item 6>

<item 7>
Life of Pi
Parker names the cub Thirsty after his enthusiasm when drinking from a nearby river . The paperwork that accompanies the shipment of the two tigers to Pi 's
family 's zoo in Pondicherry states that the cub 's name is `` Richard Parker '' and the hunter 's given name is `` Thirsty '' and his surname is `` None Given
'' , due to a mix - up with the names . Pi 's father finds the story so amusing that they continue to call the tiger `` Richard Parker '' . Reception ( edit )
Brian Bethune of Maclean 's describes Life of Pi as a `` head - scratching combination of dense religious allegory , zoological lore and enthralling adventure
tale , written with warmth and grace '' . Master Plots suggested that the `` central themes of Life of Pi concern religion and human faith in God '' . Reutter
said , `` So believable is Pi 's story telling that readers will be amazed . '' Gregory Stephens added that it `` achieves something more quietly spectacular .
'' Smith stated that there was `` no bamboozlement here . ''
</item 7>

<item 8>
Life of Pi
A recurring theme throughout the novel seems to be believability . Pi at the end of the book asks the two investigators `` If you stumble at mere believability
, what are you living for ? '' According to Gordon Houser there are two main themes of the book : `` that all life is interdependent , and that we live and
breathe via belief . '' Growth through adversity ( edit ) PBS has described Martel 's story as one of `` personal growth through adversity . '' The main
character learns that `` tigers are dangerous '' at a young age when his father forces him to watch the zoo 's Royal Bengal tiger patriarch , Mahisha , devour a
live goat . Later , after he has been reduced to eking out a desperate existence on the lifeboat with the company of a fully grown tiger , Pi develops `` alpha
'' qualities as he musters the strength , will and skills he needs to survive . Inspiration ( edit ) In a 2002 interview with PBS , Martel said `` I was sort of
looking for a story , not only with a small ' s ' but sort of with a capital ' S ' -- something that would direct my life . ''
</item 8>

<item 9>
Life of Pi
His father owns a zoo in Pondicherry . The livelihood provides the family with a relatively affluent lifestyle and some understanding of animal psychology . The
narrator describes how he acquired his full name , Piscine Molitor Patel , as a tribute to the swimming pool in France . After hearing schoolmates tease him by
transforming the first name into `` Pissing '' , he establishes the short form of his name as `` Pi '' when he starts secondary school . The name , he says ,
pays tribute to the irrational number which is the ratio of the circumference of a circle to its diameter . In recounting his experiences , Pi describes several
other unusual situations involving proper names : two visitors to the zoo , one a devout Muslim , and the other a committed atheist , bear identical names ; and
a 450 - pound tiger at the zoo bears the name Richard Parker as the result of a clerical error , in which human and animal names were reversed . Pi is raised as
a Hindu who practices vegetarianism . At the age of fourteen , he investigates Christianity and Islam , and decides to become an adherent of all three religions
, much to his parents ' dismay , saying he `` just wants to love God . ''
</item 9>

<item 10>
Life of Pi
He tries to understand God through the lens of each religion , and comes to recognize benefits in each one . A few years later in 1977 , during the period when
Indian Prime Minister Indira Gandhi declares `` The Emergency '' , Pi 's father decides to sell the zoo and immigrate with his wife and sons to Canada . Part
two ( edit ) The second part of the novel begins with Pi 's family aboard the Tsimtsum , a Japanese freighter that is transporting animals from their zoo to
North America . A few days out of port from Manila , the ship encounters a storm and sinks . Pi manages to escape in a small lifeboat , only to learn that the
boat also holds a spotted hyena , an injured Grant 's zebra , and an orangutan named Orange Juice . Much to the boy 's distress , the hyena kills the zebra and
then Orange Juice . A tiger has been hiding under the boat 's tarpaulin : it 's Richard Parker , who had boarded the lifeboat with ambivalent assistance from Pi
himself some time before the hyena attack . Suddenly emerging from his hideaway , Richard Parker kills and eats the hyena .
</item 10>
=== END ITEM id=9fb8a96ac4250804 ===

=== ITEM id=a7312da79ce009f2 ===
Gold sentences and facts:
Sentence 0: The nineteenth season of Law & Order : Special Victims Unit premiered on September 27 , 2017 .
  Fact 0.0: The nineteenth season of Law & Order: Special Victims Unit premiered.
  Fact 0.1: The nineteenth season of Law & Order: Special Victims Unit premiered on September 27, 2017.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
The show also briefly ran on Syfy in 2006 . International broadcast ( edit ) In Australia , Law & Order : SVU airs on Network Ten and Australia 's cable channel
Universal Channel ( formerly aired on TV1 which ceased to broadcast in December 2013 ) . In the UK `` Law & Order : SVU '' originally aired on the Hallmark
Channel ( which in 2012 became the Universal Channel ) with a terrestrial TV airing on Channel 5 in 2016 which then switched to Five USA . The UK is two and
half weeks behind the US . In Brazil , Law & Order : SVU also airs on Universal Channel , with weekdays episodes from past seasons and simultaneously , two
episodes after the original airing in the US . Law & Order : SVU is broadcast at 9.30 pm on Tuesdays on TV3 in New Zealand . Series 18 premiered on 11 April
2017 . References ( edit ) Notes ( edit ) Jump up ^ SVU Writers Room ( @ SVUWritersRoom ) ( 7 July 2016 ) . `` First cast read - through of # SVUSeason18 '' (
Tweet ) -- via Twitter .
</item 1>

<item 2>
Law & Order: Special Victims Unit
; least in Colorado and Utah '' . Note : Each U.S. network television season starts in late September and ends in late May ( with the exception of the second
and tenth season ) , which coincides with the completion of May sweeps . Season Time slot ( ET ) Episodes First aired Last aired TV season Rank Viewers ( in
millions ) Date Viewers ( in millions ) Date Viewers ( in millions ) Monday 9 : 00 pm ( Episodes 1 -- 9 ) Friday 10 : 00 pm ( Episodes 10 -- 22 ) 22 September
20 , 1999 ( 1999 - 09 - 20 ) 18.43 May 19 , 2000 ( 2000 - 05 - 19 ) 15.11 1999 -- 2000 No. 33 12.18 Friday 10 : 00 pm 21 October 20 , 2000 ( 2000 - 10 - 20 )
13.20 May 11 , 2001 ( 2001 - 05 - 11 ) 15.06 2000 -- 01 No.
</item 2>

<item 3>
Law & Order: Special Victims Unit
On May 12 , 2017 , the series was renewed for a nineteenth season , which premiered on September 27 , 2017 . As of January 17 , 2018 , Law & Order : Special
Victims Unit has aired 421 original episodes . Contents ( hide ) 1 Production 1.1 History and development 1.1. 1 Executive producers 1.2 Filming 1.3 Broadcast
history 1.4 Russian adaptation 2 Casting 2.1 Salaries 3 Cast and characters 4 Series overview 5 Episodes 6 Reception 6.1 U.S. television ratings 6.2 Awards and
honors 7 Broadcast and streaming 7.1 Syndication 8 International broadcast 9 References 9.1 Notes 9.2 Bibliography 10 External links Production ( edit ) History
and development ( edit ) The idea for Law & Order : Special . Victims Unit originated with the 1986 `` preppie murder '' case of Robert Chambers , who strangled
Jennifer Levin , a woman he dated whom he later killed during what he claimed was consensual `` rough sex '' in Manhattan 's Central Park .
</item 3>

<item 4>
Law & Order: Special Victims Unit
ET for the first nine episodes , from September 20 through November 29 , 1999 . It was then shifted to Friday nights at 10 p.m. ET on January 7 , 2000 , and
remained in that time slot through the end of season four on May 16 , 2003 . Beginning with the season five premiere on September 23 , 2003 , SVU moved to
Tuesday nights at 10 : 00 p.m. ET . NBC airs repeats on Saturday nights at varying times and previous episodes are shown on the USA Network on varying days in
marathon blocks With the season eleven premiere on September 23 , 2009 , the series vacated its Tuesday 10 p.m. ET slot because NBC began a prime - time
weeknight Jay Leno series . The new time slot became Wednesday nights at 9 : 00 p.m. ET on NBC , with CTV still airing SVU on Tuesdays at 10 : 00 in Canada .
After the 2010 Winter Olympics on March 3 , 2010 , the time slot for SVU changed again to Wednesdays at 10 p.m.
</item 4>

<item 5>
Law & Order: Special Victims Unit
The first episode , `` Payback '' , premiered on NBC on September 20 , 1999 . Executive producers ( edit ) Executive producer Neal Baer left Law & Order : SVU
as showrunner at the end of season twelve , after eleven years ( seasons 2 -- 12 ) on the show , in order to sign a three - year deal with CBS Studios . Baer
was replaced by former Law & Order : Criminal Intent showrunner Warren Leight . In March 2015 , it was announced that Warren Leight signed a three - year deal
with Sony Pictures Television , that will allow him to work on SVU one more season , its seventeenth . Leight joined the show in season thirteen . It was
announced on March 10 , 2016 that original Law & Order veteran producer Rick Eid would take Leight 's place as showrunner starting in season 18 . Creator Dick
Wolf commented to The Hollywood Reporter , `` I 'm extremely pleased that Rick had decided to rejoin the family and hope that he will be here for years to come
. ''
</item 5>

<item 6>
Law & Order: Special Victims Unit
2017 ( 2017 - 09 - 27 ) TBA TBA TBA Reception ( edit ) U.S. television Ratings ( edit ) Law & Order : Special Victims Unit premiered on Monday , September 20 ,
1999 . After nine episodes , the show was moved to Friday nights , where it found its audience and following its first and second seasons became a Top 20 -
overall show , overcoming the `` Friday night death slot '' phenomenon . Beginning with the season five , the show aired on Tuesdays to compete with CBS '
Judging Amy and ABC 's NYPD Blue . In its later years , Law & Order : Special Victims Unit consistently outperformed Law & Order in the Nielsen ratings for
first run episodes until the latter 's cancellation in 2010 . In 2016 , a New York Times study of the 50 TV shows with the most Facebook Likes found that SVU 's
popularity was `` atypical : generally slightly more popular in rural areas and the Black Belt , but largely restricted to the eastern half of the country . It
's most popular in Albany , N.Y. ; least in Colorado and Utah '' .
</item 6>

<item 7>
Law & Order: Special Victims Unit
Jump up ^ Ausiello , Michael ( April 11 , 2011 ) . `` Scoop : NBC Taps New Law & Order : SVU Boss '' . TVLine . Jump up ^ Littleton , Cynthia ( March 13 , 2015
) . `` ' Law & Order : SVU ' Showrunner Warren Leight Sets Overall Deal with Sony TV ( EXCLUSIVE ) '' . Variety . Retrieved May 11 , 2015 . Jump up ^ Stanhope ,
Kate ( March 10 , 2016 ) . `` ' Law & Order : SVU ' Taps Flagship Series Vet as New Showrunner '' . The Hollywood Reporter . Retrieved March 20 , 2016 . Jump up
^ Andreeva , Nellie ( May 12 , 2017 ) . `` ' Law & Order : SVU ' Showrunner Rick Eid to Run ' Chicago P.D. ' Next Season '' . Deadline.com . Retrieved May 13 ,
2017 . Jump up ^ Otterson , Joe ( May 25 , 2017 ) .
</item 7>

<item 8>
Law & Order: Special Victims Unit
During post-production of season 18 , following the announcement that SVU was renewed for a nineteenth season , it was revealed that Rick Eid departed the
series . He will be taking over another Dick Wolf / NBC series . Chicago P.D. It was announced on May 25 , 2017 that original Law & Order and Law & Order :
Criminal Intent showrunner Michael S. Chernuchin would be reprising his role starting on season nineteen . Chernuchin was also co-creator and executive
producing showrunner of Chicago Justice , another Wolf - related show that was canceled by NBC at the end of the 2016 -- 17 TV season . Filming ( edit ) SVU
shooting on location in Central Park at night Many exterior scenes of Law & Order : Special Victims Unit are filmed on location in New York City , Wolf 's
hometown , throughout all five of New York City 's boroughs . As the NYPD encounters varied law enforcement challenges on a daily basis , the backdrop provides
the writers a supply of ideal locations from which to choose . Fort Lee , New Jersey served as the filming location for Detective Elliot Stabler 's residence in
Queens , New York .
</item 8>

<item 9>
Law & Order: Special Victims Unit
1999 ( 1999 - 09 - 20 ) -- present Chronology Related shows Law & Order franchise External links Website Law & Order : Special Victims Unit ( often abbreviated
to Law & Order : SVU or just SVU ) is an American police procedural , legal , crime drama television series set in New York City , where it is also primarily
produced . In the style of the original Law & Order , episodes are often `` ripped from the headlines '' or loosely based on real crimes that have received
media attention . Created by Dick Wolf , the series premiered on NBC on September 20 , 1999 , as the second series in Wolf 's successful Law & Order franchise .
Law & Order : Special Victims Unit has been nominated for and won numerous awards , including the 2006 Emmy for Outstanding Lead Actress in a Drama Series for
Mariska Hargitay , the first , and to date only , Emmy to be received by a regular on any Law & Order series . It is the current longest running scripted
non-animated U.S.
</item 9>

<item 10>
Law & Order: Special Victims Unit
ET , where it stayed until the twelfth season . In the 12th season , SVU moved back to 9 : 00 p.m. to lead in the newest Law & Order spinoff , Law & Order : LA
, until it was pulled from the network in January 2011 to be retooled . SVU moved back to 10 : 00 p.m. on January 12 , 2011 , until the end of the 13th season .
With season 14 , SVU moved back to 9 : 00 p.m. after a two - hour season premiere event on September 26 , 2012 , a position the show has held since . Russian
adaptation ( edit ) In 2007 , the Russian production company Studio 2B purchased the rights to create an adaptation of Law & Order : Special Victims Unit for
Russian television . Titled Закон и порядок : отдел оперативных расследований ( Law & Order : Division of Field Investigation ) , the series stars Alisa Bogart
as Major Olga Bobrova .
</item 10>
=== END ITEM id=a7312da79ce009f2 ===

=== ITEM id=1f462107d88971b8 ===
Gold sentences and facts:
Sentence 0: The Ring of Fire is a direct result of plate tectonics : the movement and collisions of lithospheric plates . The eastern section of the ring is the result of the Nazca Plate and the Cocos Plate being subducted beneath the westward - moving South American Plate . The Cocos Plate is being subducted beneath the Caribbean Plate , in Central America . A portion of the Pacific Plate and the small Juan de Fuca Plate are being subducted beneath the North American Plate . Along the northern portion , the northwestward - moving Pacific plate is being subducted beneath the Aleutian Islands arc . Farther west , the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan . The southern portion is more complex , with a number of smaller tectonic plates in collision with the Pacific plate from the Mariana Islands , the Philippines , Bougainville , Tonga , and New Zealand ; this portion excludes Australia , since it lies in the center of its tectonic plate . Indonesia lies between the Ring of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt along the south and west from Sumatra , Java , Bali , Flores , and Timor . The famous and very active San Andreas Fault zone of California is a transform fault which offsets a portion of the East Pacific Rise under southwestern United States and Mexico . The motion of the fault generates numerous small earthquakes , at multiple times a day , most of which are too small to be felt . The active Queen Charlotte Fault on the west coast of the Haida Gwaii , British Columbia , has generated three large earthquakes during the 20th century : a magnitude 7 event in 1929 ; a magnitude 8.1 in 1949 ( Canada 's largest recorded earthquake ) ; and a magnitude 7.4 in 1970 .
  Fact 0.0: The Ring of Fire is a direct result of plate tectonics.
  Fact 0.1: Plate tectonics involves the movement and collisions of lithospheric plates.
  Fact 0.2: The eastern section of the Ring of Fire is the result of the Nazca Plate being subducted beneath the westward-moving South American Plate.
  Fact 0.3: The eastern section of the Ring of Fire is the result of the Cocos Plate being subducted beneath the westward-moving South American Plate.
  Fact 0.4: The Cocos Plate is being subducted beneath the Caribbean Plate in Central America.
  Fact 0.5: A portion of the Pacific Plate is being subducted beneath the North American Plate.
  Fact 0.6: The small Juan de Fuca Plate is being subducted beneath the North American Plate.
  Fact 0.7: Along the northern portion, the northwestward-moving Pacific plate is being subducted beneath the Aleutian Islands arc.
  Fact 0.8: Farther west, the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan.
  Fact 0.9: The southern portion of the Ring of Fire is more complex.
  Fact 0.10: In the southern portion, a number of smaller tectonic plates are in collision with the Pacific plate from the Mariana Islands, the Philippines, Bougainville, Tonga, and New Zealand.
  Fact 0.11: The southern portion of the Ring of Fire excludes Australia.
  Fact 0.12: Australia is excluded from the southern portion since it lies in the center of its tectonic plate.
  Fact 0.13: Indonesia lies between the Ring of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt.
  Fact 0.14: The Alpide belt runs along the south and west of Indonesia from Sumatra, Java, Bali, Flores, and Timor.
  Fact 0.15: The San Andreas Fault zone of California is famous and very active.
  Fact 0.16: The San Andreas Fault zone of California is a transform fault.
  Fact 0.17: The San Andreas Fault offsets a portion of the East Pacific Rise under southwestern United States and Mexico.
  Fact 0.18: The motion of the San Andreas Fault generates numerous small earthquakes, at multiple times a day.
  Fact 0.19: Most of the small earthquakes generated by the San Andreas Fault are too small to be felt.
  Fact 0.20: The Queen Charlotte Fault is active.
  Fact 0.21: The Queen Charlotte Fault is on the west coast of the Haida Gwaii, British Columbia.
  Fact 0.22: The Queen Charlotte Fault has generated three large earthquakes during the 20th century.
  Fact 0.23: One of the large Queen Charlotte Fault earthquakes was a magnitude 7 event in 1929.
  Fact 0.24: One of the large Queen Charlotte Fault earthquakes was a magnitude 8.1 event in 1949, which is Canada's largest recorded earthquake.
  Fact 0.25: One of the large Queen Charlotte Fault earthquakes was a magnitude 7.4 event in 1970.

Retrieved items:
<item 1>
The Pacific Ring of Fire is a roughly 40,000-km horseshoe-shaped zone of trenches, volcanic arcs, earthquakes and plate boundaries around the Pacific. Its 452
volcanoes represent over 75% of the world’s active and dormant volcanoes. About 90% of global earthquakes and 81% of the largest occur there; nearly all of the
largest volcanic eruptions of the past 11,700 years also occurred within it. The activity results from plate movement and subduction
</item 1>

<item 2>
The Ring of Fire overview lists volcanic regions from the Andes through Central America, North America, Russia, Japan, the Philippines, Indonesia, New Zealand,
and Antarctica. In Bolivia, active volcanoes lie mainly in the western Cordillera Occidental and belong to the Central Volcanic Zone, formed by Nazca Plate
subduction; ancient volcanic remnants occur in the east. Chile monitors its volcanoes through SERNAGEOMIN. Its volcanism and earthquakes are linked to Nazca
Plate sub
</item 2>

<item 3>
Ring of Fire
The southern portion is more complex , with a number of smaller tectonic plates in collision with the Pacific plate from the Mariana Islands , the Philippines ,
Bougainville , Tonga , and New Zealand ; this portion excludes Australia , since it lies in the center of its tectonic plate . Indonesia lies between the Ring
of Fire along the northeastern islands adjacent to and including New Guinea and the Alpide belt along the south and west from Sumatra , Java , Bali , Flores ,
and Timor . The famous and very active San Andreas Fault zone of California is a transform fault which offsets a portion of the East Pacific Rise under
southwestern United States and Mexico . The motion of the fault generates numerous small earthquakes , at multiple times a day , most of which are too small to
be felt . The active Queen Charlotte Fault on the west coast of the Haida Gwaii , British Columbia , has generated three large earthquakes during the 20th
century : a magnitude 7 event in 1929 ; a magnitude 8.1 in 1949 ( Canada 's largest recorded earthquake ) ; and a magnitude 7.4 in 1970 .
</item 3>

<item 4>
Ring of Fire
Islands Nauru Northern Mariana Islands Wake Island Polynesia Easter Island Hawaiian Islands Cook Islands French Polynesia Austral Islands Gambier Islands
Marquesas Islands Society Islands Tuamotu Kermadec Islands Mangareva Islands Samoa Tokelau Tonga Tuvalu Ring of Fire Regions of South America East Amazon basin
Atlantic Forest Caatinga Cerrado North Caribbean South America West Indies Los Llanos The Guianas Amazon basin Amazon rainforest Gulf of Paria Paria Peninsula
Paraguaná Peninsula Orinoco Delta South Tierra del Fuego Patagonia Pampas Pantanal Gran Chaco Chiquitano dry forests Valdes Peninsula West Andes Tropical Andes
Wet Andes Dry Andes Pariacaca mountain range Altiplano Atacama Desert Latin Hispanic American Cordillera Ring of Fire LAC Polar regions Antarctic Antarctic
Peninsula East Antarctica West Antarctica Eklund Islands Ecozone Extreme points Islands Arctic Arctic Alaska British Arctic Territories Canadian Arctic
Archipelago Finnmark Greenland Northern Canada Northwest Territories Nunavik Nunavut Russian Arctic Sakha Sápmi Yukon North American Arctic Earth 's oceans and
seas Arctic Ocean Amundsen Gulf Barents Sea Beaufort Sea Chukchi Sea East
</item 4>

<item 5>
The southern Ring of Fire is complex: smaller plates collide with the Pacific Plate from the Mariana Islands and Philippines through Bougainville and Tonga to
New Zealand; Australia is excluded because it lies within its own plate. Indonesia lies between the Ring of Fire and Alpide belt, spanning New Guinea to
Sumatra–Timor. The San Andreas transform fault offsets the East Pacific Rise and causes frequent small earthquakes. The Queen Charlotte Fault produced magnitude
7 (1929), 8.1 (
</item 5>

<item 6>
The Pacific Ring of Fire is a roughly 40,000-km horseshoe of plate boundaries, trenches, earthquakes, and volcanoes surrounding the Pacific. Its 452 volcanoes
comprise over 75% of the world’s active and dormant volcanoes; it produces about 90% of earthquakes and 81% of the largest. Subduction drives much of its
activity, though some regions form through rifting or hotspots. From the Andes and Cascades to Japan, Indonesia, and
</item 6>

<item 7>
Ring of Fire
Land areas ( edit ) Papua New Guinea and tectonic plates : Pacific Plate , Australian Plate , Caroline Plate , Banda Sea Plate ( as `` Mer de Banda '' ) ,
Woodlark Plate , Bird 's Head Plate , Maoke Plate , Solomon Sea Plate , North Bismarck Plate , South Bismarck Plate and Manus Plate ( in French ) Puysegur
Trench and Macquarie Ridge Taupo Volcanic Zone Kermadec Islands Tonga Islands Fiji Islands Bougainville and Solomon Islands New Hebrides Arc Bismarck Volcanic
Arc , junction Tanimbar and Kai Islands Lesser Sunda Islands Sunda Arc Andaman and Nicobar Islands Izu - Bonin - Mariana Arc Mariana Islands Bonin Islands Izu
Islands Philippine Mobile Belt Taiwan Ryukyu Islands Japan , Boso Triple Junction Kuril Islands Kamchatka Peninsula American Cordillera North American
Cordillera Aleutian Arc Aleutian Islands Aleutian Range Eastern Alaska , U.S. Northern Cordilleran Volcanic Province Cascade Volcanic Arc and Rio Grande rift
Sierra Nevada ( U.S. )
</item 7>

<item 8>
Ring of Fire
Trans - Mexican Volcanic Belt Central America Volcanic Arc Andes North Volcanic Zone Central Volcanic Zone South Volcanic Zone Austral Volcanic Zone Scotia Arc
South Sandwich Islands Antarctica Antarctic Peninsula Victoria Land See also ( edit ) Earthquakes portal Volcanoes portal 2004 Indian Ocean earthquake and
tsunami 2010 Chile earthquake 2011 Tōhoku earthquake and tsunami Andesite line Geology of the Pacific Northwest Katsuhiko Ishibashi Pacific Rim References (
edit ) Jump up ^ `` Ring of Fire -- Pacific Ring of Fire '' . Geography.about.com. 2010 - 06 - 14 . Retrieved 2010 - 11 - 01 . Jump up ^ `` Ring of Fire '' .
USGS. 2012 - 07 - 24 . Retrieved 2013 - 06 - 13 . Jump up ^ `` Where do earthquakes occur ? '' . USGS. 2013 - 05 - 13 . Archived from the original on 2014 - 08
- 05 . Retrieved 2013 - 06 - 13 .
</item 8>

<item 9>
Ring of Fire
Ring of Fire - wikipedia Ring of Fire Jump to : navigation , search For other uses , see Ring of Fire ( disambiguation ) . The Pacific Ring of Fire Tectonic
plates of the world Great ( magnitude 8.0 + ) earthquakes 1900 -- 2017 , with the majority positioned along the Ring of Fire The Ring of Fire is a major area in
the basin of the Pacific Ocean where a large number of earthquakes and volcanic eruptions occur . In a 40,000 km ( 25,000 mi ) horseshoe shape , it is
associated with a nearly continuous series of oceanic trenches , volcanic arcs , and volcanic belts and plate movements . It has 452 volcanoes ( more than 75 %
of the world 's active and dormant volcanoes ) . The Ring of Fire is sometimes called the circum - Pacific belt . About 90 % of the world 's earthquakes and 81
% of the world 's largest earthquakes occur along the Ring of Fire .
</item 9>

<item 10>
Ring of Fire
The next most seismically active region ( 5 -- 6 % of earthquakes and 17 % of the world 's largest earthquakes ) is the Alpide belt , which extends from Java to
the northern Atlantic Ocean via the Himalayas and southern Europe . All but three of the world 's 25 largest volcanic eruptions of the last 11,700 years
occurred at volcanoes in the Ring of Fire . The Ring of Fire is a direct result of plate tectonics : the movement and collisions of lithospheric plates . The
eastern section of the ring is the result of the Nazca Plate and the Cocos Plate being subducted beneath the westward - moving South American Plate . The Cocos
Plate is being subducted beneath the Caribbean Plate , in Central America . A portion of the Pacific Plate and the small Juan de Fuca Plate are being subducted
beneath the North American Plate . Along the northern portion , the northwestward - moving Pacific plate is being subducted beneath the Aleutian Islands arc .
Farther west , the Pacific plate is being subducted along the Kamchatka Peninsula arcs on south past Japan .
</item 10>
=== END ITEM id=1f462107d88971b8 ===
