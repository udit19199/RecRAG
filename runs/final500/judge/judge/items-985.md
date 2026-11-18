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

=== ITEM id=863635f31d9a56c7 ===
Gold sentences and facts:
Sentence 0: The Eagles headed into their bye week with the best record in the NFL ( 8 -- 1 ) , and with Wentz leading the NFL MVP discussion with an NFL - leading 23 touchdowns to 5 interceptions . During their bye , the Eagles signed former Giants offensive tackle Will Beatty and former Saints linebacker Dannell Ellerbe to not only add depth but also players with Super Bowl experiences . Coming off of the bye week , the Philadelphia Eagles entered a Sunday night showdown with the defending NFC East champion Dallas Cowboys . The Eagles came out sluggish and trailed 9 -- 7 at the half . The team exploded in the second half though , rattling off 30 unanswered points to soundly defeat their rival by a final score of 37 -- 9 . The Eagles were rolling on a nine - game winning streak after a huge blowout win over the Chicago Bears . The Eagles traveled to Seattle in Week 13 in a huge Sunday Night showdown against Russell Wilson and the Seattle Seahawks . The Eagles struggled throughout the whole game , blowing many scoring opportunities and allowing Wilson to throw 3 touchdowns . The Eagles lost 24 -- 10 , ending their win streak . However , the Eagles bounced back in Week 14 win over the Los Angeles Rams in a tough 43 -- 35 win . However , Carson Wentz left the game in the third quarter with a knee injury . It was later found out that Wentz tore his ACL , ending his MVP caliber season . Backup Nick Foles would once again take over as starting quarterback . In Foles 's first start against the New York Giants , he would lead them back from a 20 -- 7 deficit and score 4 touchdowns and win the game 34 -- 29 . Foles struggled in the last two games of the season against the Oakland Raiders and the Dallas Cowboys , and threw a touchdown and 2 interceptions in that span . Despite this , the Eagles finished 13 -- 3 and secured the top seed in the NFC . The Eagles opened as the underdogs , the first time in history that a No. 1 seed has opened up the postseason as an underdog . Despite this , Foles would lead the Eagles past the Atlanta Falcons in the Divisional Round 15 -- 10 . In the next game Foles and the Eagles annihilated the Minnesota Vikings 38 -- 7 in the NFC Championship nicknamed the Minneapolis Massacre , mocking the Minneapolis Miracle from their previous playoff victory . Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots .
  Fact 0.0: The Eagles headed into their bye week with the best record in the NFL, at 8-1.
  Fact 0.1: Carson Wentz was leading the NFL MVP discussion.
  Fact 0.2: Wentz had an NFL-leading 23 touchdowns.
  Fact 0.3: Wentz had 5 interceptions.
  Fact 0.4: During their bye, the Eagles signed former Giants offensive tackle Will Beatty.
  Fact 0.5: During their bye, the Eagles signed former Saints linebacker Dannell Ellerbe.
  Fact 0.6: The signings added depth to the Eagles.
  Fact 0.7: Beatty and Ellerbe have Super Bowl experience.
  Fact 0.8: Coming off the bye week, the Eagles entered a Sunday night showdown with the defending NFC East champion Dallas Cowboys.
  Fact 0.9: The Eagles came out sluggish against the Cowboys and trailed 9-7 at the half.
  Fact 0.10: In the second half, the Eagles scored 30 unanswered points.
  Fact 0.11: The Eagles defeated the Cowboys 37-9.
  Fact 0.12: The Eagles were on a nine-game winning streak.
  Fact 0.13: The nine-game winning streak came after a huge blowout win over the Chicago Bears.
  Fact 0.14: The Eagles traveled to Seattle in Week 13 for a Sunday Night showdown against Russell Wilson and the Seattle Seahawks.
  Fact 0.15: The Eagles struggled throughout that game, blowing many scoring opportunities.
  Fact 0.16: Wilson threw 3 touchdowns in that game.
  Fact 0.17: The Eagles lost to Seattle 24-10.
  Fact 0.18: The loss ended the Eagles' win streak.
  Fact 0.19: The Eagles bounced back in Week 14 with a tough 43-35 win over the Los Angeles Rams.
  Fact 0.20: Carson Wentz left the game in the third quarter with a knee injury.
  Fact 0.21: It was later found that Wentz tore his ACL.
  Fact 0.22: The ACL tear ended Wentz's MVP-caliber season.
  Fact 0.23: Backup Nick Foles would once again take over as starting quarterback.
  Fact 0.24: In Foles's first start against the New York Giants, he led the Eagles back from a 20-7 deficit.
  Fact 0.25: In that game, Foles scored 4 touchdowns.
  Fact 0.26: The Eagles won that game 34-29.
  Fact 0.27: Foles struggled in the last two games of the season against the Oakland Raiders and the Dallas Cowboys.
  Fact 0.28: Across those two games, Foles threw a touchdown and 2 interceptions.
  Fact 0.29: Despite this, the Eagles finished 13-3.
  Fact 0.30: The Eagles secured the top seed in the NFC.
  Fact 0.31: The Eagles opened the postseason as underdogs.
  Fact 0.32: It was the first time in history that a No. 1 seed opened the postseason as an underdog.
  Fact 0.33: Foles led the Eagles past the Atlanta Falcons 15-10 in the Divisional Round.
  Fact 0.34: In the next game, Foles and the Eagles annihilated the Minnesota Vikings 38-7 in the NFC Championship.
  Fact 0.35: That NFC Championship game was nicknamed the Minneapolis Massacre.
  Fact 0.36: The nickname mocked the Minneapolis Miracle from the Eagles' previous playoff victory.
  Fact 0.37: Foles had his best game since week 15 in the NFC Championship.
  Fact 0.38: Foles threw for 352 passing yards in that game.
  Fact 0.39: Foles threw 3 touchdowns in that game, including an impressive flea flicker touchdown pass.
  Fact 0.40: The win marked the franchise's third Super Bowl appearance.
  Fact 0.41: The win gave the Eagles a berth in Super Bowl LII.
  Fact 0.42: Super Bowl LII was a rematch of 2005 against Tom Brady and the New England Patriots.
Sentence 1: Playoff record ( all - time ) : 22 -- 21 ( last appearance after beating New England Patriots in Super Bowl LII )
  Fact 1.0: The Eagles' all-time playoff record is 22-21.
  Fact 1.1: The record is as of their last appearance, after beating the New England Patriots in Super Bowl LII.

Retrieved items:
<item 1>
Philadelphia Eagles
The team consistently ranks in the top three in attendance and has sold out every game since the 1999 season . In a Sports Illustrated poll of 321 NFL players ,
Eagles fans were selected the most intimidating fans in the NFL . Contents ( hide ) 1 Franchise history 1.1 1931 -- 1960 1.2 1961 -- 1975 1.3 1976 -- 1984 1.4
1985 -- 1993 1.5 Lurie era ( 1994 -- present ) 1.5. 1 Andy Reid era ( 1999 -- 2012 ) 1.5. 1.1 2004 Season and Super Bowl XXXIX 1.5. 1.2 2005 - 2012 1.5. 2 Chip
Kelly era ( 2013 -- 2015 ) 1.5. 2.1 Doug Pederson era ( 2016 -- present ) 1.5. 3 2016 1.5. 3.1 2017 : First Super Bowl championship 2 Season records 3 Rivalries
3.
</item 1>

<item 2>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 2>

<item 3>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 3>

<item 4>
Philadelphia Eagles
Philadelphia Eagles . September 9 , 2016 . Retrieved September 17 , 2017 . Jump up ^ `` Philadelphia Eagles Team Capsule '' ( PDF ) . 2017 Official National
Football League Record and Fact Book . National Football League . August 22 , 2017 . Retrieved September 17 , 2017 . Jump up ^ `` Top 10 NFL Rivalries Of All
Time : No. 4 Giants - Eagles '' . Sports Illustrated . December 15 , 2005 . Retrieved September 17 , 2017 . Jump up ^ Chadiha , Jeffri ( October 31 , 2007 ) .
`` Ranking the NFL 's best rivalries : Where does Colts - Pats fit ? '' . ESPN.com . Retrieved April 12 , 2008 . Jump up ^ Bryan , Dave ( September 20 , 2016 )
. `` After 8 Straight Losses , Steelers Looking For Philadelphia Freedom Sunday Against Eagles '' . SteelersDepot.com . Retrieved September 17 , 2017 .
</item 4>

<item 5>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 5>

<item 6>
Philadelphia Eagles
`` Eagles -- Giants among top rivalries '' . Philadelphia Inquirer . p . D6 . Jump up ^ `` Eagles search ends with Vermeil '' . St Petersburg Times . February 9
, 1976 . Retrieved November 5 , 2013 . Jump up ^ `` The World 's Most Valuable Sports Teams No. 11 Philadelphia Eagles '' . Forbes.com . July 12 , 2011 .
Retrieved February 11 , 2013 . Jump up ^ Heath , John . `` Denver 's Brian Dawkins : An Ageless Wolverine '' . BroncoTalk . Retrieved 24 April 2012 . Jump up ^
Dan Gelston ( Oct 26 , 2008 ) . `` Westbrook Helps Eagles Soar Above Falcons , Win 500th Game '' . NBC 10 Philadelphia . Retrieved February 11 , 2013 . Jump up
^ `` Vick , Eagles agree to 2 - year deal '' . ESPN.com . August 14 , 2009 . Retrieved February 11 , 2013 .
</item 6>

<item 7>
Philadelphia Eagles
ESPN.com placed Eagles fans fourth in the league in its 2008 survey , citing the connection between the team 's performance and the mood of the city . The last
home game which was blacked out on television in the Philadelphia market as a result of not being sold out was against the Arizona Cardinals on Sunday ,
September 12 , 1999 , which was Andy Reid 's first home game as new head coach of the Eagles . The studies note that -- win or lose -- Eagles fans can be
counted on to pack their stadium . As of August 2008 , the team had sold out 71 consecutive games , and 70,000 were on the team 's waiting list for season
tickets . Despite finishing with a 6 -- 10 record in the 2005 season , the Eagles ranked second in the NFL in merchandise sales , and single - game tickets for
the next season were sold out minutes after phone and Internet lines opened . Eagles fans have also been known to chant the famous , `` E-A-G-L-E-S -- Eagles !
! ''
</item 7>

<item 8>
Super Bowl XXXIX
The next year , the Patriots , having faltered down the stretch , entered those playoffs as the second seed behind the Broncos , who then defeated New England
in the AFC Title Game that year en route to a title of their own in Super Bowl 50 . Two years later , the Patriots , defending champions themselves from Super
Bowl LI , returned to repeat their title only to lose the aforementioned Super Bowl LII . Final statistics ( edit ) Sources : NFL.com Super Bowl XXXIX , Super
Bowl XXXIX Play Finder NE , Super Bowl XXXIX Play Finder Phi Statistical comparison ( edit ) New England Patriots Philadelphia Eagles First downs 21 24 First
downs rushing 6 First downs passing 14 18 First downs penalty Third down efficiency 4 / 12 9 / 16 Fourth down efficiency 0 / 0 0 / 0 Net yards rushing 112 45
Rushing attempts 28 17 Yards per rush 4.0 2.
</item 8>

<item 9>
Philadelphia Eagles
As a result , the Eagles and Steelers no longer played each other every year . Currently they are scheduled to meet once every four years in the regular season
, the most recent meeting being in 2016 at Philadelphia , with the Eagles winning 34 - 3 . The Steelers have lost nine straight games on the road against the
Eagles dating back to 1966 , which was also the start of the Super Bowl era . The Eagles lead the all - time series 47 -- 28 -- 3 . Logo and uniforms See also :
Uniform ( American football ) and footnote about Eagles ' uniform numbers . For several decades , the Eagles ' colors were kelly green , silver , and white . In
1954 , the Eagles , along with the Baltimore Colts , became the second team ever in the NFL to put a logo on their helmets , with silver wings on a kelly green
helmet . In 1969 , the team wore two helmet versions : Kelly green with white wings in road games , and white with kelly green wings at home . From 1970 to ' 73
, they wore the white helmets with Kelly green wings exclusively before switching back to Kelly green helmets with silver wings .
</item 9>

<item 10>
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 10>
=== END ITEM id=863635f31d9a56c7 ===

=== ITEM id=53d8b86a98be6adc ===
Gold sentences and facts:
Sentence 0: Yadava ( Golla )
  Fact 0.0: The text lists Golla in parentheses after Yadava.

Retrieved items:
<item 1>
Andhra Pradesh Legislative Assembly election, 2014
basha TDP 246 Kodur Srinivasulu 247 Rayachoti G. Sreekanth 248 Pulivendula YS Jaganmohan Reddy 249 Kamalapuram Raveendranath Reddy 250 Jammalamadugu Adinarayana
Reddy 251 Proddatur Rachamallu Siva Prasad Reddy 252 Mydukur Raghurami Reddy Kurnool 253 Allagadda Bhuma Akhila Priya 254 Srisailam Budda Rajaskar Reddy 255
Nandikotkur Aijayya 256 Kurnool S.V.
</item 1>

<item 2>
List of Telugu castes
Overseas Development Institute . p. 12 . Jump up ^ Vaugier - Chatterjee , Anne ( 2012 ) , `` Two Dominant Castes : The Socio - political System in Andhra
Pradesh '' , in Jaffrelot , Christophe ; Kumar , Sanjay , Rise of the Plebeians ? : The Changing Face of the Indian Legislative Assemblies , Routledge , p. 280
, ISBN 978 - 1 - 136 - 51662 - 7 Jump up ^ `` Central List Of Other Backward Classes '' . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_Telugu_castes&oldid=825334364 '' Categories : Indian castes Telugu society Social groups of Andhra Pradesh
Social groups of Telangana Hidden categories : Wikipedia indefinitely semi-protected pages Talk View source Contents About Wikipedia తెలుగు Edit links This page
was last edited on 12 February 2018 , at 20 : 29 . About Wikipedia
</item 2>

<item 3>
Andhra Pradesh Legislative Assembly election, 2014
Kodali Sri Venkateswara Rao YSRCP 192 Kaikalur Kamineni Srinivasa Rao BJP 193 Pedana Kagitha Venkat Rao TDP 194 Machilipatnam Kollu Ravindra 195 Avanigadda
Mandali Buddha Prasad 196 Pamarru ( SC ) Uppuleti Kalpana YSRCP 197 Penamaluru Bodepudi Prasad TDP 198 Vijayawada West Jallel Khan TDP 199 Vijayawada Central
Bonda Umamaheswara Rao TDP 200 Vijayawada East Gadde Ramamohan 201 Mylavaram Devineni Uma Maheswara Rao 202 Nandigama ( SC ) Thangirala Sowmya 203 Jaggayyapeta
Sreeram Rajagopal ( Tataiah ) Guntur 204 Pedakurapadu Sridhar 205 Tadikonda ( SC ) Tenali Sravana Kumar 206 Mangalagiri Aalla Ramakrishna Reddy YSRCP 207
Ponnuru Dhulipalla Narendra Kumar
</item 3>

<item 4>
List of Telugu castes
List of telugu castes - wikipedia List of telugu castes Jump to : navigation , search Telugu society is nowadays officially classified into a number of groups ,
some of which align with the concept of caste . Contents ( hide ) 1 Forward Classes 2 Backward Classes 3 See also 4 References Forward Classes Forward castes do
not qualify for government reservations . Those present in Andhra Pradesh are : Brahmin Kamma Kapu Komati Kshatriya Reddy Velama Some communities within these
groups are instead classified under other categories , as noted below . Backward Classes Some communities are classified as Other Backward Classes due to social
, financial , educational , and / or political discrimination . Agnikulakshatriya Palli Vadabalija Bestha Jalari Gangavar Gangaputra Goondla Vanyakulakshtriya
Vannekapu Vannereddi Pallikapu Pallireddi Nayyala Pattapu Balasanthu , Bahurupi Bandara Budabukkala Rajaka , Chakali ,
</item 4>

<item 5>
List of Telugu castes
Bahurupi Bandara Budabukkala Rajaka , Chakali , Vannar Dasari ( formerly engaged in Bhikshatana ) Dommara Gangiredlavaru Jangam whose traditional occupation is
begging Jogi Katipapala Korcha , Koracha Medari or Mahendra Mondivaru Mondibanda Banda Nayi - Brahmin ( Mangali , Mangala , Bhajanthri ) Vamsha Raj Pamula
Pardhi ( Mirshikari , Nirshikari ) Pambala Peddammavandlu , Devaravandlu , Yellammavandlu , Mutyalammavandlu Veeramushti ( Neetikotala ) , Veerabhadreeya
Valmiki Boya ( Boya Bedar , Kirataka Nishadi , Yellappi , Yellapu / Yellapondlu , Pedda Boya ) Talayari , Chunduvallu Gudala Kanjara - Bhatta Kinthala Kalinga ,
Buragana Kalinga , Buragam Kalinga , Pandiri Kalinga ,
</item 5>

<item 6>
List of Telugu castes
Mutrasi Tenugollu Munnurukapu ( Telangana ) Nagavasam ( Nagavamsa ) Neeli ( Nelli ) Pollinativelama ( of Srikakulam & Visakhapatnam Districts ) Poosala Passi
Rangarez or Bhavasara Kshatriya Sadhuchetty Satani ( Chattadasrivaishnava , Chatadi ) Tammali Turupu Kapu and Gajula Kapu of Srikakulam , Vizianagram &
Visakhapatnam Districts who are subject to social customs of divorce & remarriages among their women Uppara or Sagara Vanjara ( Vanjari ) Yadava ( Golla ) Pala
- Ekari Ayyaraka Nagaralu Pondara Lodh , Lodha , Lodhi Kurakula Kasi kapadi Gajula Balija ( who are traditionally associated with selling of Bangles ) Vannia
Vanniar Vannikula - Kshatriya Sarollu Atirasa ( of Polavaram , Gopalapuram , Koyyalagudem , Buttayagudem ,
</item 6>

<item 7>
List of Telugu castes
Puragiri Kshatriya ) Nessi or Kurni Padmasali ( Sali , Saliyan , Pattusali , Senapathulu , Thogata Sali ) Srisayana ( Segidi ) Swakulasali Thogata , Thogati or
Thogata Veerakshatriya Viswabrahmin or Viswakarma ( Ausula or Kamsali , Kammari , Kanchari , Vadla or Vadra or Vadrangi and Silpi ) Scheduled Castes converts to
Christianity and their progeny Agaru Arekatika , Katika , Quresh ( Muslim Butchers ) Atagara Bhataraju Chippolu ( Mera ) Gavara Godaba Hatkar Jakkala Jingar
Kandra Koshti Kachi Surya Balija ( Kalavanthula ) , Ganika Krishnabhalija ( Dasari , Bukka , Bukka Ayavar ) Koppulavelama Mathura Mali ( where they are not
Scheduled Tribe ) Mudiraj ,
</item 7>

<item 8>
Andhra Pradesh Legislative Assembly election, 2014
( SC ) P David raju 222 Darsi Shiddha Raghava Rao TDP 223 Parchur Y SAMBHA SIVA 224 Addanki G Ravi Kumar YSRCP 225 Chirala A Krishnamohan NP 226
Santhanuthalapadu ( SC ) Adimulapau Suresh YSRCP 227 Ongole Damacharla Janardhana Rao TDP 228 Kandukur Pothula Ramarao YSRCP 229 Kondapi ( SC ) DBV Swamy TDP
230 Markapuram Janke Venkata Reddy YSRCP 231 Giddalur Muttumula Ashok Reddy 232 Kanigiri Kadiri Babu rao TDP Nellore 233 Kavali PRATHAP KUMAR REDDY YSRCP 234
Atmakur Mekapati Gautam Reddy 235 Kovur Polamreddy srinivasulu reddy TDP 236 Nellore City P Anil Kumar Yadav YSRCP 237 Nellore Rural K Srihar Reddy 238
Sarvepalli Kakani
</item 8>

<item 9>
List of Telugu castes
Buragana Kalinga , Buragam Kalinga , Pandiri Kalinga , Kalinga Kepmare or Reddika Mondepatta Mondipatta Nokkar Pariki Muggula Yata Chopemari Kaikadi
Joshinandiwala Odde Oddilu Vaddi Vaddelu Mandula Mehtar ( Muslim ) Kunapuli Achukatlavandlu Aryakshatriya , Chittari , Giniyar , Chitrakara , Nakhas Devanga
Ediga Gowda ( Gamalla , Kalalee ) Goundla Settibalija ( of Visakhapatnam , East Godavari , West Godavari and Krishna District ) Dudekula , Laddaf , Pinjari or
Noorbash Gandla , Telikula , Devathilakula Jandra Kummara or Kulala or Salivahana Karikalabhakthulu Kaikolan or Kaikola ( Sengundam or Sengunther )
Karnabhakthulu Kuruba or Kuruma Nagavaddilu Neelakanthi Patkar ( Khatri ) Perika ( Perike Balija ,
</item 9>

<item 10>
List of Telugu castes
Gopalapuram , Koyyalagudem , Buttayagudem , Chagallu Mandals of West Godavari Distt. and Devipattanam , Korukonda and Gokavaram Mandals of East Godavari
District ) Patra Sikligar Siddula Gudia / Gudiya ( confined to Srikakulam , Vizianagaram and Visakhapatnam Districts only ) Kurmi ( confined to Telangana Region
and also Krishna District only ) Budubunjala / Bhunjwa / Bhadbhunja ( confined to Hyderabad and Ranga Reddy Districts only ) Lakkamari Kapu ( confined to
Telangana Region only ) See also Scheduled Castes and Scheduled Tribes List of Scheduled Tribes in India # Andhra Pradesh References Jump up ^ Srinivasulu , K.
`` Caste , Class and Social Articulation in Andhra Pradesh : Mapping Differential Regional Trajectories '' ( PDF ) . Overseas Development Institute . p. 12 .
</item 10>
=== END ITEM id=53d8b86a98be6adc ===

=== ITEM id=1063bae482905c99 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him . `` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied about Robin Hood 's soul having been destroyed . Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental . Belle lives on Hook 's ship ; in a deal with Gold , the Evil Queen will not harm her or the baby . Emma tells Archie about her visions and notes that Regina is n't present in them and could be the hooded figure . 114 `` The Other Shoe '' Steve Pearlman Jane Espenson & Jerome Schwartz October 9 , 2016 ( 2016 - 10 - 09 ) 4.11 In the past in the Enchanted Forest , Cinderella meets Prince Thomas at the ball , but flees when she believes he loves her stepsister Clorinda ; Clorinda actually plans to elope with Thomas ' footman , Jacob , but her mother , Lady Tremaine , abducts her to the Land of Untold Stories after forcing Cinderella to betray Clorinda 's location . In Storybrooke , Ashley hopes to reconcile with Clorinda . Aided by the Evil Queen , Tremaine lures Ashley into a trap for her and Jacob . Ashley protects Jacob ; Clorinda reunites with him and forgives Cinderella , while Tremaine is arrested . Emma struggles to control her magic . Whale helps Jekyll recreate his serum , while the Evil Queen and Hyde form an alliance and she releases him . Snow White wants normality , but David secretly investigates his father 's death ; Gold provides information in exchange for David delivering a message to Belle . 115 `` Strange Case '' Alrick Riley David H. Goodman & Nelson Soler October 16 , 2016 ( 2016 - 10 - 16 ) 3.53 In the past in Victorian England , Jekyll petitions the Academy of Science for membership but Dr. Lydgate rejects him . Rumplestiltskin arrives and magically perfects Jekyll 's serum ; Jekyll transforms into Hyde , who blackmails Lydgate for Jekyll 's membership . Mary is Lydgate 's daughter and Jekyll 's unrequited love ; she and Hyde fall in love . Jekyll accidentally fatally defenestrates Mary ; he frames Hyde . In Storybrooke , Gold imprisons Belle on the ship to protect her from Hyde ; but it is Jekyll who tries to kill Belle in revenge for Rumplestiltskin 's role in Mary 's death . Hook saves Belle by killing Jekyll , and Hyde dies of identical wounds ; Regina asks Emma to kill her if necessary to stop the Evil Queen . Gold warns Belle that their child will need his protection . Mary Margaret resumes teaching , and her assistant Shirin inspires her to draw on her life as Snow . Shirin is secretly Princess Jasmine , working with the Oracle to search for Aladdin . 116 5 `` Street Rats '' Norman Buckley Edward Kitsis & Adam Horowitz October 23 , 2016 ( 2016 - 10 - 23 ) 3.40 In the past in Agrabah , Aladdin seeks a magical weapon for Jasmine , to save the kingdom from Jafar ; he is the weapon and a Savior . Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death . Aladdin rejects Jafar and saves Jasmine . In Storybrooke , the Oracle is murdered . Emma helps Jasmine , hoping Aladdin 's survival ensures her own . Having faked his death , Aladdin reveals himself to Emma and confesses that he doomed Agrabah by using the shears , which he gives to her ; she asks Hook to dispose of them , but he secretly keeps them . The Evil Queen impersonates Archie to learn of Emma 's visions , and convinces Zelena to turn Archie back into a cricket . 117 6 `` Dark Waters '' Robert Duncan Andrew Chambliss & Brigitte Hales October 30 , 2016 ( 2016 - 10 - 30 ) 3.06 During the Dark Curse , Captain Nemo abducts Hook and invites him to join the crew `` family '' of his submarine , Nautilus , in their quest for the `` Mysterious Island '' , which is the Land of Untold Stories . Hook discovers the first mate is his half - brother , Liam , and decides to leave to avoid confrontation ; but Liam attacks him , instead mortally wounding Nemo , who steps in . In the present , Archie is freed . Henry confronts Hook over lying about the shears , and Liam imprisons them on Nautilus . Hook opens up and Henry accepts him as family ; they save each other and dispose of the shears . Modern medicine saves Nemo ; he , Liam , and Hook make amends . Emma and Aladdin discuss being a Savior . Jasmine reveals that Agrabah vanished . Snow supports Belle at her ultrasound . The Evil Queen seduces Gold and offers the retrieved shears in exchange for help taking Snow 's heart . 118 7 `` Heartless '' Ralph Hemecker Jane Espenson November 6 , 2016 ( 2016 - 11 - 06 ) 3.56 In the past , fugitive Snow and shepherd David separately travel to Longbourn ; she to flee the kingdom and he to sell his struggling farm . Without ever seeing each other 's faces , David saves Snow from a bounty hunter and she repays him with money to save the farm , deciding not to flee . Their hands touch and a magical sapling is created . In the present , the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water unless Mary Margaret and David hand over their hearts . They learn the sapling could stop the Queen , who destroys it just after they find it . But it has shown them their previous meeting , and they conclude that their love can endure anything ; they surrender themselves . The Queen casts a sleeping curse on their hearts ; because they share Snow 's heart , the curse affects them both ; each can wake the other but immediately succumbs to the curse , so they can never be awake together . Regina uses Gold 's relationship with the Queen to separate the latter from Zelena , who earns Gold 's enmity by exposing his schemes to Belle . 119 8 `` I 'll Be Your Mirror '' Jennifer Lynch Jerome Schwartz & Leah Fong November 13 , 2016 ( 2016 - 11 - 13 ) 3.40 Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror , but she traps them there , where the Dragon is already imprisoned . Henry prepares for a date with Violet ; the Queen impersonates Regina and tries to manipulate him but he recognizes her . Emma and Regina discover Sidney was building a mirror portal out of the world and try and finish it , but the Evil Queen forces the Dragon to attack Emma and Regina , using his heart , and tries to make Henry kill the Dragon with the Hammer of Hephaestus ; he instead uses it to free his mothers . Mary Margaret and David adjust to living in shifts . Belle asks Zelena to help her escape Storybrooke via magical portal , and they form an alliance with Aladdin and Jasmine . Aladdin steals the Genie 's lamp and the Sorcerer 's wand from Gold 's shop , but Gold takes the wand back from Zelena and puts a magical tracking device on Belle . He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H. Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina continue to feud . Emma obtains the sword from her vision . 121 10 `` Wish You Were Here '' Ron Underwood Edward Kitsis & Adam Horowitz December 4 , 2016 ( 2016 - 12 - 04 ) 3.27 The Black Fairy abducts Gideon . Emma discovers the sword can kill the Evil Queen without hurting Regina . The Queen steals the lamp and wishes Emma to an alternate reality where she was never the Savior ; Regina follows her . David and his allies take the lamp from the Queen , who is turned into a serpent by the hooded figure , revealed to be the adult Gideon . Jasmine wishes herself and Aladdin to Agrabah . In the `` wish realm '' , Emma lives as a princess until Regina traumatically restores her memory . Regina frees Rumplestiltskin in exchange for a magic bean , but the portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood . 122 11 `` Tougher Than the Rest '' Billy Gierhart Edward Kitsis & Adam Horowitz March 5 , 2017 ( 2017 - 03 - 05 ) 3.03 In 1990 Minnesota , a young Emma is told by a teenager that she has the power to change her fate . In present day , after they miss their way out of the Wish realm , Emma and Regina find another option for escape when they come across August , who agrees to help create a new portal . Regina goes to find Robin Hood to see if he would 've been better off without her . After Regina finds Robin , they 're caught by the Sheriff of Nottingham and later by the Wish realm Rumple . Robin believes Regina when she tells him about their being in an alternate reality . Emma discovers that August was the teenager she met back in Minnesota and inspires him to complete the portal . Emma , Regina , and Robin return to Storybrooke , where Gideon explains to Gold and Belle his intention to kill Emma so that he can acquire her savior powers and kill the Black Fairy , a move that has David furious and Belle concerned . Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12 ) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried about whether David will give him his blessing to marry Emma , but David does . Meanwhile , Regina tries to get the Wish Realm Robin assimilated into Storybrooke , but soon questions why he agreed to follow her , as it turns out that he had something else planned after he steals a chest from the vault . The truth as to who really murdered Robert is revealed when August gives Hook missing pages he took from Henry 's book . When Hook sees an illustration of David 's dad , he recognizes him and recalls murdering him during a robbery in the Enchanted Forest so that there was no witness to his crime . 124 13 `` Ill - Boding Patterns '' Ron Underwood Andrew Chambliss & Dana Horgan March 19 , 2017 ( 2017 - 03 - 19 ) 2.71 During the First Ogres War , Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old . After every last soldier had been slain , Beowulf surrenders to death just as Rumplestiltskin intervenes , killing all the ogres . Hailed as a hero in the village , Beowulf is irked by Rumple 's new title . He creates a ruse to lead Rumple to his demise . Fearing he would not be able to follow through on protecting the villagers without Dark Magic , Rumple bestows his dagger upon Baelfire as an insurance policy . Following a turbulent exchange and threat to frame Rumplestiltskin as a murderer , Beowulf retreats . Feeling the pressure of his threat , Bae commands his father to kill him . Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence . In Storybrooke , Rumple recognizes a familiar pattern in Gideon , who makes strides in his quest to kill Emma Swan at the cost of The Blue Fairy 's vitality . Hook worries about the repercussions of telling Emma his newly unveiled secret . Meanwhile , Emma unknowingly forces a proposal after happening upon Hook 's engagement ring . Robin of Locksley attempts to flee the town , but to no avail . The Evil Queen escapes her reptilian confinement and indoctrinates Robin . 125 14 `` Page 23 '' Kate Woods David H. Goodman & Brigitte Hales March 26 , 2017 ( 2017 - 03 - 26 ) 2.85 In the Enchanted Forest , The Evil Queen is in search of a spell that will help her end Snow White for good , but her father has love in mind for his daughter and leads her on a quest to Cupid 's Arrow . However , The Evil Queen rebrands the arrow to follow hatred and discovers that the person she despises the most is herself . In Storybrooke , The Evil Queen and Regina face off in a final confrontation . Just when Regina wins the upper hand , she remembers the Evil Queen 's self - loating and evenly splits the light and the darkness in their hearts , to share their love and hate and allow the Evil Queen to reform . After Emma calls off the engagement , Hook decides to board the Nautilus with Captain Nemo . Hook later changes his mind following a conversation with Snow but , before he can return , Gideon hijacks the vessel and sends it out of Storybrooke . 126 15 `` A Wondrous Place '' Steve Pearlman Jane Espenson & Jerome Schwartz April 2 , 2017 ( 2017 - 04 - 02 ) 2.80 At a loss for hope , Regina suggests a `` girls ' night out '' for Emma with her and Snow . While at the bar , she confides in the mixologist and sheds a tear . Meanwhile , the Nautilus is transported to the Enchanted Forest , where Aladdin and Jasmine board . They have yet to locate Agrabah and are growing weary . After Hook and Jasmine confide in one another , the vessel is irrevocably damaged and Jasmine uses a genie wish to relocate the group to the island where Jafar is located . They discover a cabana belonging to Ariel and Prince Eric . Ariel has collected a genie lamp that houses Jafar . Jasmine is able to sway him into divulging the location of the hidden Agrabah before defeating him with magic dust . Through true love 's kiss , Aladdin and Jasmine , along with Hook and Ariel , are transported to the restored city and Aladdin 's genie curse is broken . Ariel supplies Hook with a means to reach Emma . Emma receives the message and Gideon , no longer masquerading as the mixologist , threatens that he will use her tears to keep Hook from returning to Storybrooke unless she helps him kill the Black Fairy . Flashbacks to Ariel and Jasmine 's first meeting show the princess falter under Jafar 's trickery . 127 16 `` Mother 's Little Helper '' Billy Gierhart Story by : Edward Kitsis & Adam Horowitz Teleplay by : Paul Karp April 9 , 2017 ( 2017 - 04 - 09 ) 2.60 In Storybrooke , Gold reveals to Emma and Snow that the Black Fairy is his mother and that the Dark Curse Regina cast was created by the Black Fairy . Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is controlling him to kill Emma . 128 17 `` Awake '' Sharat Raju Andrew Chambliss & Leah Fong April 16 , 2017 ( 2017 - 04 - 16 ) 2.51 Regina works feverishly to wake Snow and Charming from the Sleeping Curse . Gold comes face to face with the Black Fairy , the powerful enemy Emma must fight in the Final Battle . Hook , still in Neverland , meets up with Tiger Lily , who gives him a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy . Gold confronts the Black Fairy over her manipulating Gideon with his heart and she maintains that Storybrooke would be destroyed if the two of them fought . A pixie flower allows Emma to open a doorway to Neverland and rescue Hook , who then re-proposes properly . The townsfolk gather around Emma 's sleeping parents and each take small amounts of the Sleeping Curse unto themselves , diluting it and allowing Snow and David to finally reunite . In 1993 , a pixie flower awakens Snow and Charming from their cursed selves , but an awakened Rumplestiltskin tells them they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior . 129 18 `` Where Bluebirds Fly '' Michael Schultz David H. Goodman & Brigitte Hales April 23 , 2017 ( 2017 - 04 - 23 ) 2.69 In Zelena 's past , her childhood friend Stanum asks for her help in finding the Crimson Heart , a magical artifact that would be able to restore his human form after having been cursed by the Wicked Witch of the North to slowly turn into tin . Zelena 's refusal to give up her magic leads to her betraying Stanum , as she keeps the Heart and leaves him to become the Tin Man . In the Present , The Black Fairy tempts Zelena with an offer to join her , only to have Zelena oppose her instead . Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm . In the Present , as everyone looks for the missing wand , Gold brings Gideon and Emma into the Dream World to search for Gideon 's heart , where the events of the flashbacks unfold . Gold then confronts Fiona , but she reveals the real reason behind why he was abandoned . Gold later retrieves Gideon 's heart and tells everyone that the Black Fairy is dead , paving the way for Emma and Hook 's wedding . However , Gold and Fiona make a deal that will see the Final Battle go forward , with Emma being the Savior that will fight in it . Meanwhile , Regina helps Zelena adjust to life without her powers . 131 20 `` The Song in Your Heart '' Ron Underwood David H. Goodman & Andrew Chambliss May 7 , 2017 ( 2017 - 05 - 07 ) 2.87 In the past , Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength , which also spreads to Oz as well . When Regina becomes affected by the wish , she finds a way to end it before the first curse takes place , but the Blue Fairy , who granted Snow the wish , tells Snow and David that the wish will be part of Emma 's destiny . In Emma 's past , her chance to make use of her talent is held back by being told she 'll be alone . This would come back to taunt Emma in the present day , as Fiona returns to use that weakness to take Emma 's heart before unleashing a new curse on her wedding day , with help from Gold by freezing her family . When Henry discovers the page in the book that reveals the song in Emma 's heart will make her stronger , not only does it embolden Emma , it stops Fiona from crushing her heart , but Fiona still tells Emma that the battle will continue . On her wedding day , Emma and Hook finally marry , just in time for the dark clouds of the Black Fairy 's curse to erupt from the clock tower and engulf them at 6 : 00 PM , setting up the final battle . 132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle . Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their `` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity . Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
  Fact 0.0: Episode 112, 'The Savior', was directed by Eagle Egilsson and written by Edward Kitsis and Adam Horowitz.
  Fact 0.1: Episode 112 aired on September 25, 2016 and had 3.99 million U.S. viewers.
  Fact 0.2: In the past, Aladdin, a 'Savior', is taunted by Jafar over the inevitable downfall of all Saviors.
  Fact 0.3: In Storybrooke, refugees arrive from the Land of Untold Stories.
  Fact 0.4: Emma experiences tremors and visions.
  Fact 0.5: An arrested Hyde helps Emma receive an oracle's vision of a hooded figure killing her.
  Fact 0.6: Guided by Morpheus, Gold enters Belle's dreams to lift the sleeping curse.
  Fact 0.7: Gold tries to renew his love with Belle, but she rejects him.
  Fact 0.8: 'Morpheus' reveals he is really Belle and Gold's unborn son, and that he desired this outcome.
  Fact 0.9: Belle wakes and leaves Gold.
  Fact 0.10: Henry convinces Regina that Hades may have lied about Robin Hood's soul having been destroyed.
  Fact 0.11: Zelena moves in with Regina.
  Fact 0.12: After learning her sister Regina blames her for Robin Hood's death, Zelena moves out.
  Fact 0.13: Zelena is approached by the Evil Queen after moving out.
  Fact 0.14: Episode 113, 'A Bitter Draught', was directed by Ron Underwood and written by Andrew Chambliss and Dana Horgan.
  Fact 0.15: Episode 113 aired on October 2, 2016 and had 3.72 million U.S. viewers.
  Fact 0.16: In the past in the Enchanted Forest, the Evil Queen offers to aid the revenge plot of Edmond Dantes, The Count of Monte Cristo.
  Fact 0.17: The Evil Queen's offer to Edmond Dantes is conditional on him killing Snow and David.
  Fact 0.18: Edmond Dantes falls in love with Snow's handmaiden.
  Fact 0.19: Edmond Dantes reneges on the revenge plot and flees with the handmaiden to the Land of Untold Stories to save her from poison.
  Fact 0.20: In Storybrooke, the Evil Queen recruits Zelena.
  Fact 0.21: In Storybrooke, the Evil Queen uses Edmond's heart to set him against Snow and David.
  Fact 0.22: Regina kills Edmond to save the others.
  Fact 0.23: Regina realizes she retains the capacity for darkness.
  Fact 0.24: The Evil Queen declares that the undesirable 'untold stories' will be played out.
  Fact 0.25: The Evil Queen gives David evidence that his father's death was not accidental.
  Fact 0.26: Belle lives on Hook's ship.
  Fact 0.27: In a deal with Gold, the Evil Queen will not harm Belle or the baby.
  Fact 0.28: Emma tells Archie about her visions.
  Fact 0.29: Emma notes that Regina is not present in her visions and could be the hooded figure.
  Fact 0.30: Episode 114, 'The Other Shoe', was directed by Steve Pearlman and written by Jane Espenson and Jerome Schwartz.
  Fact 0.31: Episode 114 aired on October 9, 2016 and had 4.11 million U.S. viewers.
  Fact 0.32: In the past in the Enchanted Forest, Cinderella meets Prince Thomas at the ball.
  Fact 0.33: Cinderella flees when she believes Thomas loves her stepsister Clorinda.
  Fact 0.34: Clorinda actually plans to elope with Thomas' footman, Jacob.
  Fact 0.35: Clorinda's mother, Lady Tremaine, abducts Clorinda to the Land of Untold Stories.
  Fact 0.36: Lady Tremaine abducts Clorinda after forcing Cinderella to betray Clorinda's location.
  Fact 0.37: In Storybrooke, Ashley hopes to reconcile with Clorinda.
  Fact 0.38: Aided by the Evil Queen, Lady Tremaine lures Ashley into a trap for her and Jacob.
  Fact 0.39: Ashley protects Jacob.
  Fact 0.40: Clorinda reunites with Jacob and forgives Cinderella.
  Fact 0.41: Lady Tremaine is arrested.
  Fact 0.42: Emma struggles to control her magic.
  Fact 0.43: Whale helps Jekyll recreate his serum.
  Fact 0.44: The Evil Queen and Hyde form an alliance, and she releases him.
  Fact 0.45: Snow White wants normality.
  Fact 0.46: David secretly investigates his father's death.
  Fact 0.47: Gold provides information in exchange for David delivering a message to Belle.
  Fact 0.48: Episode 115, 'Strange Case', was directed by Alrick Riley and written by David H. Goodman and Nelson Soler.
  Fact 0.49: Episode 115 aired on October 16, 2016 and had 3.53 million U.S. viewers.
  Fact 0.50: In the past in Victorian England, Jekyll petitions the Academy of Science for membership.
  Fact 0.51: Dr. Lydgate rejects Jekyll's petition.
  Fact 0.52: Rumplestiltskin arrives and magically perfects Jekyll's serum.
  Fact 0.53: Jekyll transforms into Hyde.
  Fact 0.54: Hyde blackmails Lydgate for Jekyll's membership.
  Fact 0.55: Mary is Lydgate's daughter and Jekyll's unrequited love.
  Fact 0.56: Mary and Hyde fall in love.
  Fact 0.57: Jekyll accidentally fatally defenestrates Mary.
  Fact 0.58: Jekyll frames Hyde for Mary's death.
  Fact 0.59: In Storybrooke, Gold imprisons Belle on the ship to protect her from Hyde.
  Fact 0.60: Jekyll tries to kill Belle in revenge for Rumplestiltskin's role in Mary's death.
  Fact 0.61: Hook saves Belle by killing Jekyll.
  Fact 0.62: Hyde dies of identical wounds.
  Fact 0.63: Regina asks Emma to kill her if necessary to stop the Evil Queen.
  Fact 0.64: Gold warns Belle that their child will need his protection.
  Fact 0.65: Mary Margaret resumes teaching.
  Fact 0.66: Mary Margaret's assistant Shirin inspires her to draw on her life as Snow.
  Fact 0.67: Shirin is secretly Princess Jasmine.
  Fact 0.68: Princess Jasmine is working with the Oracle to search for Aladdin.
  Fact 0.69: Episode 116, 'Street Rats', is season episode 5 and was directed by Norman Buckley and written by Edward Kitsis and Adam Horowitz.
  Fact 0.70: Episode 116 aired on October 23, 2016 and had 3.40 million U.S. viewers.
  Fact 0.71: In the past in Agrabah, Aladdin seeks a magical weapon for Jasmine to save the kingdom from Jafar.
  Fact 0.72: Aladdin is the weapon and a Savior.
  Fact 0.73: Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death.
  Fact 0.74: Aladdin rejects Jafar and saves Jasmine.
  Fact 0.75: In Storybrooke, the Oracle is murdered.
  Fact 0.76: Emma helps Jasmine, hoping Aladdin's survival ensures her own.
  Fact 0.77: Aladdin has faked his death.
  Fact 0.78: Aladdin reveals himself to Emma.
  Fact 0.79: Aladdin confesses that he doomed Agrabah by using the shears.
  Fact 0.80: Aladdin gives the shears to Emma.
  Fact 0.81: Emma asks Hook to dispose of the shears.
  Fact 0.82: Hook secretly keeps the shears.
  Fact 0.83: The Evil Queen impersonates Archie to learn of Emma's visions.
  Fact 0.84: The Evil Queen convinces Zelena to turn Archie back into a cricket.
  Fact 0.85: Episode 117, 'Dark Waters', is season episode 6 and was directed by Robert Duncan and written by Andrew Chambliss and Brigitte Hales.
  Fact 0.86: Episode 117 aired on October 30, 2016 and had 3.06 million U.S. viewers.
  Fact 0.87: During the Dark Curse, Captain Nemo abducts Hook.
  Fact 0.88: Captain Nemo invites Hook to join the crew 'family' of his submarine, Nautilus.
  Fact 0.89: The crew is on a quest for the 'Mysterious Island', which is the Land of Untold Stories.
  Fact 0.90: Hook discovers the first mate is his half-brother, Liam.
  Fact 0.91: Hook decides to leave to avoid confrontation.
  Fact 0.92: Liam attacks Hook, instead mortally wounding Nemo, who steps in.
  Fact 0.93: In the present, Archie is freed.
  Fact 0.94: Henry confronts Hook over lying about the shears.
  Fact 0.95: Liam imprisons Henry and Hook on Nautilus.
  Fact 0.96: Hook opens up and Henry accepts him as family.
  Fact 0.97: Henry and Hook save each other and dispose of the shears.
  Fact 0.98: Modern medicine saves Nemo.
  Fact 0.99: Nemo, Liam and Hook make amends.
  Fact 0.100: Emma and Aladdin discuss being a Savior.
  Fact 0.101: Jasmine reveals that Agrabah vanished.
  Fact 0.102: Snow supports Belle at her ultrasound.
  Fact 0.103: The Evil Queen seduces Gold.
  Fact 0.104: The Evil Queen offers Gold the retrieved shears in exchange for help taking Snow's heart.
  Fact 0.105: Episode 118, 'Heartless', is season episode 7 and was directed by Ralph Hemecker and written by Jane Espenson.
  Fact 0.106: Episode 118 aired on November 6, 2016 and had 3.56 million U.S. viewers.
  Fact 0.107: In the past, fugitive Snow and shepherd David separately travel to Longbourn.
  Fact 0.108: Snow travels to Longbourn to flee the kingdom.
  Fact 0.109: David travels to Longbourn to sell his struggling farm.
  Fact 0.110: Without seeing each other's faces, David saves Snow from a bounty hunter.
  Fact 0.111: Snow repays David with money to save the farm.
  Fact 0.112: Snow decides not to flee.
  Fact 0.113: Their hands touch and a magical sapling is created.
  Fact 0.114: In the present, the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water.
  Fact 0.115: The Evil Queen's threat is unless Mary Margaret and David hand over their hearts.
  Fact 0.116: Mary Margaret and David learn the sapling could stop the Queen.
  Fact 0.117: The Queen destroys the sapling just after they find it.
  Fact 0.118: The sapling showed Mary Margaret and David their previous meeting.
  Fact 0.119: Mary Margaret and David conclude that their love can endure anything.
  Fact 0.120: Mary Margaret and David surrender themselves.
  Fact 0.121: The Queen casts a sleeping curse on their hearts.
  Fact 0.122: Because they share Snow's heart, the curse affects both of them.
  Fact 0.123: Each can wake the other but immediately succumbs to the curse.
  Fact 0.124: They can never be awake together.
  Fact 0.125: Regina uses Gold's relationship with the Queen to separate the Queen from Zelena.
  Fact 0.126: Zelena earns Gold's enmity by exposing his schemes to Belle.
  Fact 0.127: Episode 119, 'I'll Be Your Mirror', is season episode 8 and was directed by Jennifer Lynch and written by Jerome Schwartz and Leah Fong.
  Fact 0.128: Episode 119 aired on November 13, 2016 and had 3.40 million U.S. viewers.
  Fact 0.129: Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror.
  Fact 0.130: The Evil Queen traps Regina and Emma in the World Behind the Mirror.
  Fact 0.131: The Dragon is already imprisoned in the World Behind the Mirror.
  Fact 0.132: Henry prepares for a date with Violet.
  Fact 0.133: The Queen impersonates Regina and tries to manipulate Henry.
  Fact 0.134: Henry recognizes the Queen.
  Fact 0.135: Emma and Regina discover Sidney was building a mirror portal out of the world.
  Fact 0.136: Emma and Regina try to finish the mirror portal.
  Fact 0.137: The Evil Queen forces the Dragon to attack Emma and Regina, using his heart.
  Fact 0.138: The Evil Queen tries to make Henry kill the Dragon with the Hammer of Hephaestus.
  Fact 0.139: Henry instead uses the Hammer of Hephaestus to free his mothers.
  Fact 0.140: Mary Margaret and David adjust to living in shifts.
  Fact 0.141: Belle asks Zelena to help her escape Storybrooke via magical portal.
  Fact 0.142: Belle and Zelena form an alliance with Aladdin and Jasmine.
  Fact 0.143: Aladdin steals the Genie's lamp and the Sorcerer's wand from Gold's shop.
  Fact 0.144: Gold takes the wand back from Zelena.
  Fact 0.145: Gold puts a magical tracking device on Belle.
  Fact 0.146: Gold asks the Queen to kill Zelena.
  Fact 0.147: The connection between Gold's and Zelena's hearts protects Zelena from Gold.
  Fact 0.148: Episode 120, 'Changelings', is season episode 9 and was directed by Mairzee Almas and written by David H. Goodman and Brian Ridings.
  Fact 0.149: Episode 120 aired on November 27, 2016 and had 3.28 million U.S. viewers.
  Fact 0.150: In the Enchanted Forest, Rumplestiltskin, while served by Belle, abducts a baby.
  Fact 0.151: Rumplestiltskin uses the abducted baby as bait for the Black Fairy, the mother who abandoned him.
  Fact 0.152: The Black Fairy escapes without answering Rumplestiltskin's questions.
  Fact 0.153: In the present, Belle receives advice from her unborn child.
  Fact 0.154: In the present, Gold becomes increasingly unhinged.
  Fact 0.155: The Evil Queen resents Gold's order to attack Zelena.
  Fact 0.156: The Evil Queen accelerates Belle's pregnancy.
  Fact 0.157: Mother Superior delivers Belle's son, Gideon.
  Fact 0.158: Mother Superior sends Gideon into hiding.
  Fact 0.159: Gold vows to find Gideon.
  Fact 0.160: The Genie has been freed.
  Fact 0.161: Aladdin takes the Genie's place.
  Fact 0.162: Zelena and Regina continue to feud.
  Fact 0.163: Emma obtains the sword from her vision.
  Fact 0.164: Episode 121, 'Wish You Were Here', is season episode 10 and was directed by Ron Underwood and written by Edward Kitsis and Adam Horowitz.
  Fact 0.165: Episode 121 aired on December 4, 2016 and had 3.27 million U.S. viewers.
  Fact 0.166: The Black Fairy abducts Gideon.
  Fact 0.167: Emma discovers the sword can kill the Evil Queen without hurting Regina.
  Fact 0.168: The Queen steals the lamp.
  Fact 0.169: The Queen wishes Emma to an alternate reality where Emma was never the Savior.
  Fact 0.170: Regina follows the Queen into the alternate reality.
  Fact 0.171: David and his allies take the lamp from the Queen.
  Fact 0.172: The Queen is turned into a serpent by the hooded figure.
  Fact 0.173: The hooded figure is revealed to be the adult Gideon.
  Fact 0.174: Jasmine wishes herself and Aladdin to Agrabah.
  Fact 0.175: In the 'wish realm', Emma lives as a princess.
  Fact 0.176: Regina traumatically restores Emma's memory in the wish realm.
  Fact 0.177: Regina frees Rumplestiltskin in exchange for a magic bean.
  Fact 0.178: The portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood.
  Fact 0.179: Episode 122, 'Tougher Than the Rest', is season episode 11 and was directed by Billy Gierhart and written by Edward Kitsis and Adam Horowitz.
  Fact 0.180: Episode 122 aired on March 5, 2017 and had 3.03 million U.S. viewers.
  Fact 0.181: In 1990 Minnesota, a young Emma is told by a teenager that she has the power to change her fate.
  Fact 0.182: In the present day, Emma and Regina miss their way out of the Wish realm.
  Fact 0.183: Emma and Regina find another option for escape when they come across August.
  Fact 0.184: August agrees to help create a new portal.
  Fact 0.185: Regina goes to find Robin Hood to see if he would have been better off without her.
  Fact 0.186: Regina and Robin are caught by the Sheriff of Nottingham.
  Fact 0.187: Regina and Robin are later caught by the Wish realm Rumple.
  Fact 0.188: Robin believes Regina when she tells him about their being in an alternate reality.
  Fact 0.189: Emma discovers that August was the teenager she met in Minnesota.
  Fact 0.190: Emma inspires August to complete the portal.
  Fact 0.191: Emma, Regina and Robin return to Storybrooke.
  Fact 0.192: Gideon explains to Gold and Belle his intention to kill Emma.
  Fact 0.193: Gideon intends to kill Emma so that he can acquire her savior powers and kill the Black Fairy.
  Fact 0.194: Gideon's intention has David furious and Belle concerned.
  Fact 0.195: Emma and Gideon finally meet for the battle.
  Fact 0.196: Emma survives, choosing her own fate.
  Fact 0.197: Gideon retreats, proclaiming this far from over.
  Fact 0.198: Gold and Belle decide to work together for the good of their son.
  Fact 0.199: Episode 123, 'Murder Most Foul', is season episode 12 and was directed by Morgan Beggs and written by Jerome Schwartz and Jane Espenson.
  Fact 0.200: Episode 123 aired on March 12, 2017 and had 3.06 million U.S. viewers.
  Fact 0.201: A flashback details how Robert, the father of David and James, was murdered.
  Fact 0.202: In the flashback, Robert learns what happened to James after James ran away from King George.
  Fact 0.203: With help from Rumple, Robert tracks James to Pleasure Island.
  Fact 0.204: King George cuts short Robert and James' reunion.
  Fact 0.205: King George orders Robert's execution.
  Fact 0.206: Robert's death affects David in the present day.
  Fact 0.207: David sees visions of Robert and asks Hook to help him, despite Hook being reformed.
  Fact 0.208: Hook becomes concerned that David is taking the investigation too far by going after Albert.
  Fact 0.209: Hook intervenes and brings David back to sanity.
  Fact 0.210: Hook is worried about whether David will give him his blessing to marry Emma.
  Fact 0.211: David gives Hook his blessing to marry Emma.
  Fact 0.212: Regina tries to get the Wish Realm Robin assimilated into Storybrooke.
  Fact 0.213: Regina questions why Robin agreed to follow her.
  Fact 0.214: Robin had something else planned, and he steals a chest from the vault.
  Fact 0.215: The truth about who murdered Robert is revealed when August gives Hook missing pages he took from Henry's book.
  Fact 0.216: Hook sees an illustration of David's dad and recognizes him.
  Fact 0.217: Hook recalls murdering David's dad during a robbery in the Enchanted Forest so that there was no witness to his crime.
  Fact 0.218: Episode 124, 'Ill-Boding Patterns', is season episode 13 and was directed by Ron Underwood and written by Andrew Chambliss and Dana Horgan.
  Fact 0.219: Episode 124 aired on March 19, 2017 and had 2.71 million U.S. viewers.
  Fact 0.220: During the First Ogres War, Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old.
  Fact 0.221: After every last soldier had been slain, Beowulf surrenders to death just as Rumplestiltskin intervenes.
  Fact 0.222: Rumplestiltskin kills all the ogres.
  Fact 0.223: Beowulf is irked by Rumple's new title as a hero in the village.
  Fact 0.224: Beowulf creates a ruse to lead Rumple to his demise.
  Fact 0.225: Fearing he would not be able to protect the villagers without Dark Magic, Rumple bestows his dagger upon Baelfire as an insurance policy.
  Fact 0.226: Beowulf retreats after a turbulent exchange and a threat to frame Rumplestiltskin as a murderer.
  Fact 0.227: Feeling the pressure of Beowulf's threat, Bae commands his father to kill him.
  Fact 0.228: Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence.
  Fact 0.229: In Storybrooke, Rumple recognizes a familiar pattern in Gideon.
  Fact 0.230: Gideon makes strides in his quest to kill Emma Swan at the cost of the Blue Fairy's vitality.
  Fact 0.231: Hook worries about the repercussions of telling Emma his newly unveiled secret.
  Fact 0.232: Emma unknowingly forces a proposal after happening upon Hook's engagement ring.
  Fact 0.233: Robin of Locksley attempts to flee the town, but to no avail.
  Fact 0.234: The Evil Queen escapes her reptilian confinement.
  Fact 0.235: The Evil Queen indoctrinates Robin.
  Fact 0.236: Episode 125, 'Page 23', is season episode 14 and was directed by Kate Woods and written by David H. Goodman and Brigitte Hales.
  Fact 0.237: Episode 125 aired on March 26, 2017 and had 2.85 million U.S. viewers.
  Fact 0.238: In the Enchanted Forest, the Evil Queen is in search of a spell that will help her end Snow White for good.
  Fact 0.239: The Evil Queen's father has love in mind for his daughter and leads her on a quest to Cupid's Arrow.
  Fact 0.240: The Evil Queen rebrands the arrow to follow hatred.
  Fact 0.241: The Evil Queen discovers that the person she despises the most is herself.
  Fact 0.242: In Storybrooke, the Evil Queen and Regina face off in a final confrontation.
  Fact 0.243: When Regina wins the upper hand, she remembers the Evil Queen's self-loathing.
  Fact 0.244: Regina evenly splits the light and the darkness in their hearts, to share their love and hate.
  Fact 0.245: Regina's split allows the Evil Queen to reform.
  Fact 0.246: After Emma calls off the engagement, Hook decides to board the Nautilus with Captain Nemo.
  Fact 0.247: Hook later changes his mind following a conversation with Snow.
  Fact 0.248: Before Hook can return, Gideon hijacks the Nautilus.
  Fact 0.249: Gideon sends the Nautilus out of Storybrooke.
  Fact 0.250: Episode 126, 'A Wondrous Place', is season episode 15 and was directed by Steve Pearlman and written by Jane Espenson and Jerome Schwartz.
  Fact 0.251: Episode 126 aired on April 2, 2017 and had 2.80 million U.S. viewers.
  Fact 0.252: At a loss for hope, Regina suggests a 'girls' night out' for Emma with her and Snow.
  Fact 0.253: At the bar, Regina confides in the mixologist and sheds a tear.
  Fact 0.254: In the Enchanted Forest, the Nautilus is transported.
  Fact 0.255: Aladdin and Jasmine board the Nautilus.
  Fact 0.256: Aladdin and Jasmine have yet to locate Agrabah and are growing weary.
  Fact 0.257: After Hook and Jasmine confide in one another, the vessel is irrevocably damaged.
  Fact 0.258: Jasmine uses a genie wish to relocate the group to the island where Jafar is located.
  Fact 0.259: The group discovers a cabana belonging to Ariel and Prince Eric.
  Fact 0.260: Ariel has collected a genie lamp that houses Jafar.
  Fact 0.261: Jasmine sways Jafar into divulging the location of the hidden Agrabah.
  Fact 0.262: Jasmine defeats Jafar with magic dust.
  Fact 0.263: Through true love's kiss, Aladdin and Jasmine, along with Hook and Ariel, are transported to the restored city.
  Fact 0.264: Aladdin's genie curse is broken.
  Fact 0.265: Ariel supplies Hook with a means to reach Emma.
  Fact 0.266: Emma receives the message.
  Fact 0.267: Gideon, no longer masquerading as the mixologist, threatens to use Emma's tears to keep Hook from returning to Storybrooke.
  Fact 0.268: Gideon's threat is unless Emma helps him kill the Black Fairy.
  Fact 0.269: Flashbacks to Ariel and Jasmine's first meeting show the princess faltering under Jafar's trickery.
  Fact 0.270: Episode 127, 'Mother's Little Helper', is season episode 16 and was directed by Billy Gierhart, with a story by Edward Kitsis and Adam Horowitz and a teleplay by Paul Karp.
  Fact 0.271: Episode 127 aired on April 9, 2017 and had 2.60 million U.S. viewers.
  Fact 0.272: In Storybrooke, Gold reveals to Emma and Snow that the Black Fairy is his mother.
  Fact 0.273: Gold reveals to Emma and Snow that the Dark Curse Regina cast was created by the Black Fairy.
  Fact 0.274: Emma agrees to help Gideon kill the Black Fairy.
  Fact 0.275: Gideon double-crosses Emma.
  Fact 0.276: Gideon leaves Emma to die at the hands of a giant spider.
  Fact 0.277: Gideon opens a portal with the sword.
  Fact 0.278: The portal allows the Black Fairy to slip through into Storybrooke.
  Fact 0.279: In the Enchanted Forest, Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard.
  Fact 0.280: As Blackbeard and Hook go through the portal, it malfunctions thanks to Gideon.
  Fact 0.281: Blackbeard and Hook end up in Neverland.
  Fact 0.282: Blackbeard and Hook are on the run from the remnants of Peter Pan's Lost Boys.
  Fact 0.283: In Storybrooke, Henry experiences a disorienting shift in his Author powers.
  Fact 0.284: Henry and Regina seek help from the previous Author, Isaac.
  Fact 0.285: Isaac reveals to Regina and Henry that they are nearing the end of the story.
  Fact 0.286: In the story, the savior must fight the Final Battle.
  Fact 0.287: Flashbacks reveal the abuses the Black Fairy inflicted on Gideon.
  Fact 0.288: The Black Fairy ripped out Gideon's heart.
  Fact 0.289: The Black Fairy is controlling Gideon to kill Emma.
  Fact 0.290: Episode 128, 'Awake', is season episode 17 and was directed by Sharat Raju and written by Andrew Chambliss and Leah Fong.
  Fact 0.291: Episode 128 aired on April 16, 2017 and had 2.51 million U.S. viewers.
  Fact 0.292: Regina works feverishly to wake Snow and Charming from the Sleeping Curse.
  Fact 0.293: Gold comes face to face with the Black Fairy.
  Fact 0.294: The Black Fairy is the powerful enemy Emma must fight in the Final Battle.
  Fact 0.295: In Neverland, Hook meets up with Tiger Lily.
  Fact 0.296: Tiger Lily gives Hook a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy.
  Fact 0.297: Gold confronts the Black Fairy over her manipulating Gideon with his heart.
  Fact 0.298: The Black Fairy maintains that Storybrooke would be destroyed if she and Gold fought.
  Fact 0.299: A pixie flower allows Emma to open a doorway to Neverland.
  Fact 0.300: Emma rescues Hook.
  Fact 0.301: Hook re-proposes properly.
  Fact 0.302: The townsfolk gather around Emma's sleeping parents.
  Fact 0.303: The townsfolk each take small amounts of the Sleeping Curse unto themselves, diluting it.
  Fact 0.304: Diluting the curse allows Snow and David to finally reunite.
  Fact 0.305: In 1993, a pixie flower awakens Snow and Charming from their cursed selves.
  Fact 0.306: An awakened Rumplestiltskin tells Snow and Charming they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior.
  Fact 0.307: Episode 129, 'Where Bluebirds Fly', is season episode 18 and was directed by Michael Schultz and written by David H. Goodman and Brigitte Hales.
  Fact 0.308: Episode 129 aired on April 23, 2017 and had 2.69 million U.S. viewers.
  Fact 0.309: In Zelena's past, her childhood friend Stanum asks for her help in finding the Crimson Heart.
  Fact 0.310: The Crimson Heart is a magical artifact that would restore Stanum's human form.
  Fact 0.311: Stanum was cursed by the Wicked Witch of the North to slowly turn into tin.
  Fact 0.312: Zelena refuses to give up her magic.
  Fact 0.313: Zelena betrays Stanum by keeping the Heart.
  Fact 0.314: Zelena leaves Stanum to become the Tin Man.
  Fact 0.315: In the present, the Black Fairy tempts Zelena with an offer to join her.
  Fact 0.316: Zelena opposes the Black Fairy instead.
  Fact 0.317: Regina is brought into the chaos by trying to stop her sister.
  Fact 0.318: Regina and Zelena find the Black Fairy and Gideon in the crystal mines.
  Fact 0.319: The Black Fairy tricks Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle.
  Fact 0.320: Zelena realizes there is nothing left for her back in Oz.
  Fact 0.321: Zelena decides to help the residents fight back against the Black Fairy.
  Fact 0.322: Zelena uses the Crimson Heart to give up her magic.
  Fact 0.323: Giving up her magic undoes the mess Zelena made in the mines.
  Fact 0.324: Emma uses her magic and one of the crystals to bring Mother Superior out of her coma.
  Fact 0.325: Regina and Zelena help Emma bring Mother Superior out of her coma.
  Fact 0.326: Episode 130, 'The Black Fairy', is season episode 19 and was directed by Alrick Riley and written by Jerome Schwartz and Dana Horgan.
  Fact 0.327: Episode 130 aired on April 30, 2017 and had 3.05 million U.S. viewers.
  Fact 0.328: The origin of Rumplestiltskin's true destiny is revealed.
  Fact 0.329: Rumplestiltskin's mother, Fiona, learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior.
  Fact 0.330: Fiona fears for her son and chooses power over protecting Rumple.
  Fact 0.331: Fiona becomes the Black Fairy.
  Fact 0.332: The Blue Fairy banishes Fiona to the Dark Realm.
  Fact 0.333: In the present, everyone looks for the missing wand.
  Fact 0.334: Gold brings Gideon and Emma into the Dream World to search for Gideon's heart.
  Fact 0.335: The events of the flashbacks unfold in the Dream World.
  Fact 0.336: Gold confronts Fiona.
  Fact 0.337: Fiona reveals the real reason behind why Gold was abandoned.
  Fact 0.338: Gold retrieves Gideon's heart.
  Fact 0.339: Gold tells everyone that the Black Fairy is dead.
  Fact 0.340: The Black Fairy's death paves the way for Emma and Hook's wedding.
  Fact 0.341: Gold and Fiona make a deal that will see the Final Battle go forward.
  Fact 0.342: Emma will be the Savior that fights in the Final Battle.
  Fact 0.343: Regina helps Zelena adjust to life without her powers.
  Fact 0.344: Episode 131, 'The Song in Your Heart', is season episode 20 and was directed by Ron Underwood and written by David H. Goodman and Andrew Chambliss.
  Fact 0.345: Episode 131 aired on May 7, 2017 and had 2.87 million U.S. viewers.
  Fact 0.346: In the past, Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength.
  Fact 0.347: The wish also spreads to Oz.
  Fact 0.348: Regina becomes affected by the wish.
  Fact 0.349: Regina finds a way to end the wish before the first curse takes place.
  Fact 0.350: The Blue Fairy, who granted Snow the wish, tells Snow and David that the wish will be part of Emma's destiny.
  Fact 0.351: In Emma's past, her chance to make use of her talent is held back by being told she'll be alone.
  Fact 0.352: The weakness of being told she'll be alone comes back to taunt Emma in the present day.
  Fact 0.353: Fiona returns to use that weakness to take Emma's heart.
  Fact 0.354: Fiona unleashes a new curse on Emma's wedding day, with help from Gold.
  Fact 0.355: The curse freezes Emma's family.
  Fact 0.356: Henry discovers the page in the book that reveals the song in Emma's heart will make her stronger.
  Fact 0.357: The song embolden Emma.
  Fact 0.358: The song stops Fiona from crushing Emma's heart.
  Fact 0.359: Fiona tells Emma that the battle will continue.
  Fact 0.360: On her wedding day, Emma and Hook finally marry.
  Fact 0.361: Dark clouds of the Black Fairy's curse erupt from the clock tower at 6:00 PM.
  Fact 0.362: The dark clouds engulf Emma and Hook.
  Fact 0.363: The curse sets up the final battle.
  Fact 0.364: Season episodes 21 and 22, 'The Final Battle', were directed by Steve Pearlman and Ralph Hemecker and written by Edward Kitsis and Adam Horowitz.
  Fact 0.365: The Final Battle aired on May 14, 2017 and had 2.95 million U.S. viewers.
  Fact 0.366: Fiona's curse alters Storybrooke.
  Fact 0.367: Fiona's curse institutionalizes Emma.
  Fact 0.368: Fiona's curse leaves only Henry and Gold 'awake'.
  Fact 0.369: Fiona's curse sends Emma's other allies to their home realms.
  Fact 0.370: The Final Battle begins over Emma's belief.
  Fact 0.371: If Emma stops believing in the realms of story, they will be consumed by the Darkness and cease to exist.
  Fact 0.372: If all the realms are destroyed, Fiona would gain unchecked powers.
  Fact 0.373: If all the realms are destroyed, Fiona would be able to break the Laws of Magic at will.
  Fact 0.374: In the Enchanted Forest, the banished characters gather at Regina's castle.
  Fact 0.375: David and Hook obtain a magic bean.
  Fact 0.376: The Evil Queen sacrifices herself.
  Fact 0.377: The Evil Queen holds off the Darkness to delay the group's destruction.
  Fact 0.378: Fiona convinces Emma to destroy the book.
  Fact 0.379: Henry hand-draws a new book.
  Fact 0.380: Henry's hand-drawn book restores Emma's belief in time to save everyone in the castle.
  Fact 0.381: Gold kills Fiona.
  Fact 0.382: Fiona had hidden Belle and turned her into a coward.
  Fact 0.383: Gold's killing of Fiona breaks the Curse.
  Fact 0.384: Breaking the Curse restores everyone's memories.
  Fact 0.385: Breaking the Curse reunites everyone in Storybrooke.
  Fact 0.386: Gideon is still under orders to kill Emma.
  Fact 0.387: Gold attempts to free Gideon's heart.
  Fact 0.388: A spell cast by the Black Fairy prevents Gold from freeing Gideon's heart.
  Fact 0.389: Emma lays down her sword, allowing Gideon to impale her.
  Fact 0.390: Gideon impaling Emma releases Emma's light magic and ends the battle.
  Fact 0.391: Gold and Emma have both chosen good.
  Fact 0.392: Gideon is returned to infancy.
  Fact 0.393: The realms are restored.
  Fact 0.394: Those who died in the crisis are revived.
  Fact 0.395: The characters embark on their 'happy beginnings'.
  Fact 0.396: Many years later in the New Enchanted Forest, an adult Henry vanishes while defending his daughter Lucy from an unidentified entity.
  Fact 0.397: Guided by Tiger Lily, Lucy travels to Seattle.
  Fact 0.398: In Seattle, Henry no longer remembers Lucy.

Retrieved items:
<item 1>
Once Upon a Time (season 6)
Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Regular 2.2 Recurring 2.3 Guest 3 Episodes 4 Production 4.1 Development 4.1. 1 Musical episode 4.2 Casting
5 Ratings 6 References 7 External links Premise ( edit ) The residents of Storybrooke are faced with threats from the Evil Queen and Mr. Hyde , both with
different agendas following the arrival of refugees from the Land of Untold Stories . This leads to events to test Emma 's savior abilities that results in the
arrival of the Black Fairy to begin the Final Battle that was prophesied centuries ago . As dark and light collides , a new , and darker curse is unleashed ,
leading to the fall of all the realms with magic . The Final Battle is fought and won , leading to the heroes earning their happy beginnings , but a new
adventure begins in the far future for an adult Henry involving his daughter Lucy , and another curse looming across the New Enchanted Forest inhabitants .
</item 1>

<item 2>
Once Upon a Time (season 6)
Jekyll accidentally fatally defenestrates Mary ; he frames Hyde . In Storybrooke , Gold imprisons Belle on the ship to protect her from Hyde ; but it is Jekyll
who tries to kill Belle in revenge for Rumplestiltskin 's role in Mary 's death . Hook saves Belle by killing Jekyll , and Hyde dies of identical wounds ;
Regina asks Emma to kill her if necessary to stop the Evil Queen . Gold warns Belle that their child will need his protection . Mary Margaret resumes teaching ,
and her assistant Shirin inspires her to draw on her life as Snow . Shirin is secretly Princess Jasmine , working with the Oracle to search for Aladdin . 116 5
`` Street Rats '' Norman Buckley Edward Kitsis & Adam Horowitz October 23 , 2016 ( 2016 - 10 - 23 ) 3.40 In the past in Agrabah , Aladdin seeks a magical weapon
for Jasmine , to save the kingdom from Jafar ; he is the weapon and a Savior .
</item 2>

<item 3>
Once Upon a Time (season 6)
He creates a ruse to lead Rumple to his demise . Fearing he would not be able to follow through on protecting the villagers without Dark Magic , Rumple bestows
his dagger upon Baelfire as an insurance policy . Following a turbulent exchange and threat to frame Rumplestiltskin as a murderer , Beowulf retreats . Feeling
the pressure of his threat , Bae commands his father to kill him . Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his
innocence . In Storybrooke , Rumple recognizes a familiar pattern in Gideon , who makes strides in his quest to kill Emma Swan at the cost of The Blue Fairy 's
vitality . Hook worries about the repercussions of telling Emma his newly unveiled secret . Meanwhile , Emma unknowingly forces a proposal after happening upon
Hook 's engagement ring . Robin of Locksley attempts to flee the town , but to no avail . The Evil Queen escapes her reptilian confinement and indoctrinates
Robin .
</item 3>

<item 4>
Once Upon a Time (season 6)
He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H.
Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as
bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn
child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior
delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina
continue to feud . Emma obtains the sword from her vision .
</item 4>

<item 5>
Once Upon a Time (season 6)
Ashley protects Jacob ; Clorinda reunites with him and forgives Cinderella , while Tremaine is arrested . Emma struggles to control her magic . Whale helps
Jekyll recreate his serum , while the Evil Queen and Hyde form an alliance and she releases him . Snow White wants normality , but David secretly investigates
his father 's death ; Gold provides information in exchange for David delivering a message to Belle . 115 `` Strange Case '' Alrick Riley David H. Goodman &
Nelson Soler October 16 , 2016 ( 2016 - 10 - 16 ) 3.53 In the past in Victorian England , Jekyll petitions the Academy of Science for membership but Dr. Lydgate
rejects him . Rumplestiltskin arrives and magically perfects Jekyll 's serum ; Jekyll transforms into Hyde , who blackmails Lydgate for Jekyll 's membership .
Mary is Lydgate 's daughter and Jekyll 's unrequited love ; she and Hyde fall in love . Jekyll accidentally fatally defenestrates Mary ; he frames Hyde .
</item 5>

<item 6>
Once Upon a Time (season 6)
The Evil Queen escapes her reptilian confinement and indoctrinates Robin . 125 14 `` Page 23 '' Kate Woods David H. Goodman & Brigitte Hales March 26 , 2017 (
2017 - 03 - 26 ) 2.85 In the Enchanted Forest , The Evil Queen is in search of a spell that will help her end Snow White for good , but her father has love in
mind for his daughter and leads her on a quest to Cupid 's Arrow . However , The Evil Queen rebrands the arrow to follow hatred and discovers that the person
she despises the most is herself . In Storybrooke , The Evil Queen and Regina face off in a final confrontation . Just when Regina wins the upper hand , she
remembers the Evil Queen 's self - loating and evenly splits the light and the darkness in their hearts , to share their love and hate and allow the Evil Queen
to reform . After Emma calls off the engagement , Hook decides to board the Nautilus with Captain Nemo .
</item 6>

<item 7>
Once Upon a Time (season 6)
Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle
decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12
) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after
he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's
execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook
becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried
about whether David will give him his blessing to marry Emma , but David does .
</item 7>

<item 8>
Once Upon a Time (season 6)
Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter
Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to
aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and
flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him
against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the
undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental .
</item 8>

<item 9>
Once Upon a Time (season 6)
132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters
Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle
begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are
destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at
Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona
convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle .
</item 9>

<item 10>
Once Upon a Time (season 6)
Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in
Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents
him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma
having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their
`` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity .
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
</item 10>
=== END ITEM id=1063bae482905c99 ===

=== ITEM id=07555d840800d307 ===
Gold sentences and facts:
Sentence 0: Because the states were preexisting political entities , the U.S. Constitution did not need to define or explain federalism in any one section but it often mentions the rights and responsibilities of state governments and state officials in relation to the federal government . The federal government has certain express powers ( also called enumerated powers ) which are powers spelled out in the Constitution , including the right to levy taxes , declare war , and regulate interstate and foreign commerce . In addition , the Necessary and Proper Clause gives the federal government the implied power to pass any law `` necessary and proper '' for the execution of its express powers . Other powers -- the reserved powers -- are reserved to the people or the states . The power delegated to the federal government was significantly expanded by the Supreme Court decision in McCulloch v. Maryland ( 1819 ) , amendments to the Constitution following the Civil War , and by some later amendments -- as well as the overall claim of the Civil War , that the states were legally subject to the final dictates of the federal government .
  Fact 0.0: The states were preexisting political entities.
  Fact 0.1: The U.S. Constitution did not need to define or explain federalism in any one section, because the states were preexisting political entities.
  Fact 0.2: The U.S. Constitution often mentions the rights and responsibilities of state governments and state officials in relation to the federal government.
  Fact 0.3: The federal government has certain express powers, also called enumerated powers.
  Fact 0.4: Express powers are powers spelled out in the Constitution.
  Fact 0.5: The federal government's express powers include the right to levy taxes.
  Fact 0.6: The federal government's express powers include the right to declare war.
  Fact 0.7: The federal government's express powers include the right to regulate interstate commerce.
  Fact 0.8: The federal government's express powers include the right to regulate foreign commerce.
  Fact 0.9: The Necessary and Proper Clause gives the federal government the implied power to pass any law 'necessary and proper' for the execution of its express powers.
  Fact 0.10: Other powers, the reserved powers, are reserved to the people or the states.
  Fact 0.11: The Supreme Court decision in McCulloch v. Maryland (1819) significantly expanded the power delegated to the federal government.
  Fact 0.12: Amendments to the Constitution following the Civil War significantly expanded the power delegated to the federal government.
  Fact 0.13: Some later amendments to the Constitution significantly expanded the power delegated to the federal government.
  Fact 0.14: The overall claim of the Civil War, that the states were legally subject to the final dictates of the federal government, significantly expanded the power delegated to the federal government.

Retrieved items:
<item 1>
Federalism
Under the Constitution Act ( previously known as the British North America Act ) of 1867 , specific powers of legislation are allotted . Section 91 of the
constitution gives rise to federal authority for legislation , whereas section 92 gives rise to provincial powers . For matters not directly dealt with in the
constitution , the federal government retains residual powers ; however , conflict between the two levels of government , relating to which level has
legislative jurisdiction over various matters , has been a longstanding and evolving issue . Areas of contest include legislation with respect to regulation of
the economy , taxation , and natural resources . India ( edit ) Main article : Federalism in India Indian state governments led by various political parties The
Government of India ( referred to as the Union Government ) was established by the Constitution of India , and is the governing authority of a federal union of
29 states and 7 union territories . The government of India is based on a 3 tiered system , in which the Constitution of India delineates the subjects on which
each tier of government has executive powers . The Constitution originally provided for a two - tier system of government , the Union Government ( also known as
the Central Government ) , representing the Union of India , and the State governments .
</item 1>

<item 2>
Federalism
American government has evolved from a system of dual federalism to one of associative federalism . In `` Federalist No. 46 , '' James Madison asserted that the
states and national government `` are in fact but different agents and trustees of the people , constituted with different powers . '' Alexander Hamilton ,
writing in `` Federalist No. 28 , '' suggested that both levels of government would exercise authority to the citizens ' benefit : `` If their ( the peoples ' )
rights are invaded by either , they can make use of the other as the instrument of redress . '' ( 1 ) The United States is composed of fifty self - governing
states and several territories . Because the states were preexisting political entities , the U.S. Constitution did not need to define or explain federalism in
any one section but it often mentions the rights and responsibilities of state governments and state officials in relation to the federal government . The
federal government has certain express powers ( also called enumerated powers ) which are powers spelled out in the Constitution , including the right to levy
taxes , declare war , and regulate interstate and foreign commerce .
</item 2>

<item 3>
Federalism
The U.S. Constitution was written as a reaction to the Articles of Confederation , under which the United States was a loose confederation with a weak central
government . In contrast , Europe has a greater history of unitary states than North America , thus European `` federalism '' argues for a weaker central
government , relative to a unitary state . The modern American usage of the word is much closer to the European sense . As the power of the Federal government
has increased , some people have perceived a much more unitary state than they believe the Founding Fathers intended . Most people politically advocating ``
federalism '' in the United States argue in favor of limiting the powers of the federal government , especially the judiciary ( see Federalist Society , New
Federalism ) . In Canada , federalism typically implies opposition to sovereigntist movements ( most commonly Quebec separatism ) . The governments of Argentina
, Australia , Brazil , India , and Mexico , among others , are also organized along federalist principles . Federalism may encompass as few as two or three
internal divisions , as is the case in Belgium or Bosnia and Herzegovina .
</item 3>

<item 4>
Federalism
- J. Proudhon , The Principle of Federation , 1863 . A Comparative Bibliography : Regulatory Competition on Corporate Law A Rhetoric for Ratification : The
Argument of the Federalist and its Impact on Constitutional Interpretation Brainstorming National ( in French ) Teaching about Federalism in the United States -
From the Education Resources Information Center Clearinghouse for Social Studies / Social Science Education Bloomington , Indiana . An Ottawa , Ontario , Canada
- based international organization for federal countries that share best practices among countries with that system of government Tenth Amendment Center
Federalism and States Rights in the U.S. BackStory Radio episode on the origins and current status of Federalism Constitutional law scholar Hester Lessard
discusses Vancouver 's Downtown Eastside and jurisdictional justice McGill University ,
</item 4>

<item 5>
Federalism
See also ( edit ) Consociationalism Cooperative federalism Democratic World Federalists Federal Union Layer cake federalism Pillarisation States ' rights Union
of Utrecht World Federalist Movement Notes and references ( edit ) Jump up ^ In 1946 , scholar Kenneth Wheare observed that the two levels of government in the
US were ' co-equally supreme ' . In this he echoed the perspective of the founding fathers , James Madison in Federalist 39 having seen the several states as
forming ' distinct and independent portions of the supremacy ' in relation to the general government . Wheare , Kenneth ( 1946 ) Federal Government , Oxford
University Press , London , pp. 10 - 15 . Madison , James , Hamilton , Alexander and Jay , John ( 1987 ) The Federalist Papers , Penguin , Harmondsworth , p.
258 . Jump up ^ Law , John ( 2013 ) ' How Can We Define Federalism ? ' , in Perspectives on Federalism , Vol. 5 , No. 3 , pp. E105 - 6 .
</item 5>

<item 6>
Federalism
Its distinctive feature , exemplified in the founding example of modern federalism by the United States of America under the Constitution of 1787 , is a
relationship of parity between the two levels of government established . It can thus be defined as a form of government in which there is a division of powers
between two levels of government of equal status . Federalism differs from confederalism , in which the general level of government is subordinate to the
regional level , and from devolution within a unitary state , in which the regional level of government is subordinate to the general level . It represents the
central form in the pathway of regional integration or separation , bounded on the less integrated side by confederalism and on the more integrated side by
devolution within a unitary state . Leading examples of the federation or federal state include the United States , Canada , Brazil , Germany , Switzerland ,
Argentina , Australia and India . Some also today characterize the European Union as the pioneering example of federalism in a multi-state setting , in a
concept termed the federal union of states . Contents ( hide ) 1 Overview 1.1 Explanations for adoption of federalist systems 2 European vs.
</item 6>

<item 7>
Federalism
they are already constitutional , parliamentary , and administratively modernized states ) . European vs. American federalism ( edit ) Main articles : Federal
Europe and Federalism in the United States In Europe , `` Federalist '' is sometimes used to describe those who favor a common federal government , with
distributed power at regional , national and supranational levels . Most European federalists want this development to continue within the European Union .
European federalism originated in post-war Europe ; one of the more important initiatives was Winston Churchill 's speech in Zürich in 1946 . In the United
States , federalism originally referred to belief in a stronger central government . When the U.S. Constitution was being drafted , the Federalist Party
supported a stronger central government , while `` Anti-Federalists '' wanted a weaker central government . This is very different from the modern usage of ``
federalism '' in Europe and the United States . The distinction stems from the fact that `` federalism '' is situated in the middle of the political spectrum
between a confederacy and a unitary state . The U.S.
</item 7>

<item 8>
Federalism
In 1995 the Supreme Court rejected the Gun - Free School Zones Act in the Lopez decision , and also rejected the civil remedy portion of the Violence Against
Women Act of 1994 in the United States v. Morrison decision . Recently , the Commerce Clause was interpreted to include marijuana laws in the Gonzales v. Raich
decision . Dual federalism holds that the federal government and the state governments are co-equals , each sovereign . However , since the Civil War Era , the
national courts often interpret the federal government as the final judge of its own powers under dual federalism . The establishment of Native American
governments ( which are separate and distinct from state and federal government ) exercising limited powers of sovereignty , has given rise to the concept of ``
bi-federalism . '' Venezuela ( edit ) The Federal War ended in 1863 with the signing of the Treaty of Coche by both the centralist government of the time and
the Federal Forces . The United States of Venezuela were subsequently incorporated under a `` Federation of Sovereign States '' upon principles borrowed from
the Articles of Confederation of the United States of America .
</item 8>

<item 9>
Federalism
This was reaffirmed by the Tenth Amendment to the United States Constitution , which reserves all powers and rights that are not delegated to the Federal
Government as left to the States and to the people . Bicameralism ( edit ) The structures of most federal governments incorporate mechanisms to protect the
rights of component states . One method , known as ' intrastate federalism ' , is to directly represent the governments of component states in federal political
institutions . Where a federation has a bicameral legislature the upper house is often used to represent the component states while the lower house represents
the people of the nation as a whole . A federal upper house may be based on a special scheme of apportionment , as is the case in the senates of the United
States and Australia , where each state is represented by an equal number of senators irrespective of the size of its population .
</item 9>

<item 10>
Federalism
neither a single large unitary state nor a league / confederation among several small states , but a hybrid of the two ) . In the course of the nineteenth
century the meaning of federalism would come to shift , strengthening to refer uniquely to the novel compound political form , while the meaning of
confederalism would remain at a league of states . Thus , this article relates to the modern usage of the word ' federalism ' . Modern federalism is a system
based upon democratic rules and institutions in which the power to govern is shared between national and provincial / state governments . The term federalist
describes several political beliefs around the world depending on context . Federalism is sometimes viewed as in the context of international negotiation as ``
the best system for integrating diverse nations , ethnic groups , or combatant parties , all of whom may have cause to fear control by an overly powerful center
. '' However , in some countries , those skeptical of federal prescriptions believe that increased regional autonomy is likely to lead to secession or
dissolution of the nation . In Syria , federalization proposals have failed in part because `` Syrians fear that these borders could turn out to be the same as
the ones that the fighting parties have currently carved out . ''
</item 10>
=== END ITEM id=07555d840800d307 ===

=== ITEM id=6a2ff6e74449256f ===
Gold sentences and facts:
Sentence 0: Draft lotteries were conducted again in 1970 ( for those born in 1951 ) and from 1971 to 1975 ( for 1952 to 1956 births ) . The draft numbers issued in 1972 were never used the next year to call for induction into service , because the last call was December 7 and authority to induct expired on June 30 , 1973 .
  Fact 0.0: Draft lotteries were conducted again in 1970 for men born in 1951.
  Fact 0.1: Draft lotteries were conducted from 1971 to 1975 for men born from 1952 to 1956.
  Fact 0.2: The draft numbers issued in 1972 were never used the next year to call men for induction into service.
  Fact 0.3: The last call for induction was on December 7.
  Fact 0.4: Authority to induct expired on June 30, 1973.
Sentence 1: On December 1 , 1969 , the Selective Service System of the United States conducted two lotteries to determine the order of call to military service in the Vietnam War for men born from 1944 to 1950 . These lotteries occurred during a period of conscription from just before World War II to 1973 . It was the first time a lottery system had been used to select men for military service since 1942 .
  Fact 1.0: On December 1, 1969, the Selective Service System of the United States conducted two lotteries.
  Fact 1.1: The 1969 lotteries determined the order of call to military service in the Vietnam War.
  Fact 1.2: The 1969 lotteries were for men born from 1944 to 1950.
  Fact 1.3: The 1969 lotteries occurred during a period of conscription from just before World War II to 1973.
  Fact 1.4: The 1969 lotteries were the first time a lottery system had been used to select men for military service since 1942.

Retrieved items:
<item 1>
Draft lottery (1969)
The 1969 draft lottery only encouraged resentment of the Vietnam War and the draft . It strengthened the anti-war movement , and all over the United States ,
people decried discrimination by the draft system `` against low - education , low - income , underprivileged members of society '' . The draft lottery had
social and economic consequences because it generated resistance to military service and the resisters , draft evaders or `` draft dodgers '' , were generally
young , well - educated , healthy men . The fear of military service in Vietnam influenced many young men born in the late 1940s to join the National Guard .
These young men were aware that the National Guard would be unlikely to send its soldiers to Vietnam . Many men were unable to join the National Guard , even
though they had passed their physicals , because many state National Guards had long waiting lists just to enlist . Still others chose legal sanctions such as
imprisonment , either showing their disapproval by burning their draft cards or draft letters , or simply not presenting themselves for the military service
test . Others left the country , commonly moving to Canada .
</item 1>

<item 2>
Draft lottery (1969)
Others left the country , commonly moving to Canada . The number of American citizens who moved to Canada during the Vietnam war because of the draft is
estimated to be around 125,000 ; it is believed that about half returned to the United States after the Nixon era ( when the war was also over ) ( 1975 ) .
Modification ( edit ) Lottery procedure was improved the next year although public discontent continued to grow until `` authority to induct expired on June 30
, 1973 '' . In 1970 , covering 1951 birthdates for use during 1971 ( sometimes called the 1971 draft ) , scientists at the National Bureau of Standards prepared
78 random permutations of the numbers 1 to 366 using random numbers selected from published tables . From the 78 permutations , 25 were selected at random and
transcribed to calendars using 1 = January 1 , 2 = January 2 , ... 365 = December 31 . Those calendars were sealed in envelopes. 25 more permutations were
selected and sealed in 25 more envelopes without transcription to calendars . The two sets of 25 envelopes were furnished to the Selective Service System .
</item 2>

<item 3>
Draft lottery (1969)
The draft numbers issued in 1972 were never used the next year to call for induction into service , because the last call was December 7 and authority to induct
expired on June 30 , 1973 . The 1972 to 1975 lottery numbers were used to call some men born 1953 to 1956 for physical exams . The highest number called for a
physical was 215 ( for tables 1970 through 1976 ) . Origins and consequences ( edit ) The reason for the 1969 draft was to add personnel to a war to stop the
spread of communism in Vietnam . After World War II , Japan left Vietnamese Emperor Bao Dai in command . Ho Chi Minh , another Vietnamese leader , saw this as
an opportunity to strike and took over Hanoi . Ho Chi Minh then set up the Democratic Republic of Vietnam . Emperor Bo , in the South , set up the Vietnam State
with Saigon as the capital . Ho Chi Minh was basing his political structure off of the communist states such as Russia while Emperor Bao Dai in the south wanted
a Vietnam that was modeled after the west , like the United States .
</item 3>

<item 4>
Draft lottery (1969)
In 1967 , the number of U.S. military personnel in Vietnam was around 500,000 . The war was costing the U.S. $25 billion a year , and many of the young men
drafted were being sent to a war they wanted no part of . Martin Luther King Jr. also started to support the anti-war movement on the grounds of it being
immoral and the amount of African Americans that were being killed . November 15 , 1969 marked the largest anti war movement in the history of the United States
. This protest featured many anti-war political speakers and popular singers of the time . Many people at the time saw Richard Nixon as a liar ; when he took
office , he claimed that he would begin troop withdrawals from Vietnam immediately . After ten months of being in office , the president had yet to start
withdrawals , anyone and the people felt lied to . Later , president Nixon claimed to have been watching sports as the anti-war demonstration took place outside
the White House . The 1970s were a time of turmoil in the United States , beginning with the civil rights movement which set the standards for practices by the
anti-war movement .
</item 4>

<item 5>
Draft lottery (1969)
Last modified 2009 - 03 - 24 . Confirmed 2011 - 05 - 26 . -- contemporary news stories , images of Official Orders ( call to physical exam , call to report ) ,
lottery results , Draft Board classifications , Vietnam troop levels , induction statistics 1917 -- 73 . David Lane ( 2003 ) . Introduction to Graphs : Clearing
Up the Draft with Graphs . Connexions. 18 July 2003 . Confirmed 2011 - 05 - 26 . -- list of draft ranks , additional analysis . Selective Service System ( 2009
) . The Vietnam Lotteries . SSS : History and Records . Last updated 2015 - 09 - 19 . Norton Starr ( 1997 ) . `` Nonrandom Risk : The 1970 Draft Lottery '' .
Journal of Statistics Education 5.2 ( 1997 ) . Confirmed 2011 - 05 - 26 . -- The online edition includes instructions for getting the data online and a lesson
plan for statistics class using the 1970 and 1971 draft lottery data .
</item 5>

<item 6>
Draft lottery (1969)
had only committed around 800 personnel to train and outfit the South Vietnamese . In 1961 , the John F. Kennedy administration started working under the ``
Domino Theory , '' which stated that if South Vietnam was to fall to the North , then other places in southeast Asia were to become vulnerable to the communists
as well . This caused President Kennedy to begin sending additional American soldiers to Vietnam . By 1962 , there were around 9,000 personnel in Vietnam . In
1963 , a coup was organized by South Vietnamese generals which resulted in the death of Diem . President Lyndon B. Johnson increased U.S. personnel in Vietnam
due to the political instability in the country . In August 1964 , two U.S. warships were attacked by North Vietnamese torpedo boats . Johnson issued an attack
order against North Vietnam , and Congress passed a motion which gave him more authority over military decisions . In 1965 , President Johnson had sent 82,000
troops to Vietnam , and his officials wanted another 175,000 .
</item 6>

<item 7>
Draft lottery (1969)
( 1969 ) . `` The Cost of the Draft and the Cost of Ending the Draft '' . American Economic Review . 59 ( 3 ) . Jump up ^ Ifill , Gwen ( 13 February 1992 ) . ``
THE 1992 CAMPAIGN : New Hampshire ; Clinton Thanked Colonel in ' 69 For ' Saving Me From the Draft ' '' -- via NYTimes.com . ^ Jump up to : Rosenblatt , J.R. ;
Filliben , J.J. ( 1971 ) . `` Randomization and the Draft Lottery '' . Science . 171 : 306 -- 08 . doi : 10.1126 / science. 171.3968. 306 . Jump up ^ `` How the
U.S. Draft Works '' . HowStuffWorks. 2001 - 10 - 18 . Retrieved 2017 - 12 - 05 . External links ( edit ) Landscaper.net ( 2009 ) . The Military Draft and 1969
Draft Lottery for the Vietnam War . Last modified 2009 - 03 - 24 .
</item 7>

<item 8>
Draft lottery (1969)
Jump up ^ 91st U.S. Congress . `` AN ACT To amend the Military Selective Service Act of 1967 ... '' ( pdf ) . United States Government Printing Office . ( Pub.
L. 91 -- 124 , 83 Stat. 220 , enacted November 26 , 1969 ) Jump up ^ Peters , Gerhard ; Woolley , John T. `` Richard Nixon : `` Executive Order 11497 - Amending
the Selective Service Regulations to Prescribe Random Selection , '' November 26 , 1969 `` . The American Presidency Project . University of California - Santa
Barbara . ^ Jump up to : `` Vietnam War Protests - Vietnam War - HISTORY.com '' . HISTORY.com . Retrieved 2017 - 12 - 05 . Jump up ^
https://www.nytimes.com/1984/09/02/us/college-enrollment-linked-to-vietnam-war.html , https://timesmachine.nytimes.com/timesmachine/1967/03/19/107186611.pdf
Jump up ^ Fisher , Anthony C. ( 1969 ) .
</item 8>

<item 9>
Draft lottery (1969)
Due to the heavy demand for military personnel , the United States required more than what the regular military could provide , causing the acceleration of the
draft . Between 1965 and 1972 the draft provided 2,215,000 service members to the U.S. military . Previously in the United States , during the War of 1812 ,
President Madison established a commission to recommend the best ways to raise military manpower ; to keep the draft or to institute a volunteer army . After
much debate within the Nixon administration and Congress , it was decided that an all - volunteer force was affordable , feasible , and would enhance the nation
's security . President Richard Nixon issued an executive order prescribing regulations for random selection by the United States Selective Service on November
26 , 1969 . In the 1960s anti-war movements started to occur in the U.S. , mainly among those on college campuses and in more leftist circles , especially those
who embraced the `` hippie '' lifestyle . College students were entitled to a deferment ( 2 - S status ) but were subject to the draft if they dropped out or
graduated . In 1967 , the number of U.S.
</item 9>

<item 10>
Draft lottery (1969)
Draft lottery ( 1969 ) - wikipedia Draft lottery ( 1969 ) Jump to : navigation , search This article includes a list of references , but its sources remain
unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise citations . ( October 2017 ) ( Learn how
and when to remove this template message ) Representative Alexander Pirnie ( R - NY ) drawing the first number . On December 1 , 1969 , the Selective Service
System of the United States conducted two lotteries to determine the order of call to military service in the Vietnam War for men born from 1944 to 1950 . These
lotteries occurred during a period of conscription from just before World War II to 1973 . It was the first time a lottery system had been used to select men
for military service since 1942 . The lottery numbers assigned in December 1969 were used during calendar year 1970 both to call for induction and to call for
physical examination , a preliminary call covering more men .
</item 10>
=== END ITEM id=6a2ff6e74449256f ===

=== ITEM id=24f318d3f9dba1e7 ===
Gold sentences and facts:
Sentence 0: The Day the Music Died The memorial marking the crash site , 2003 Accident summary Date February 3 , 1959 ( 1959 - 02 - 03 ) Summary Spatial disorientation , loss of control in near - IMC conditions Site Near Clear Lake , Iowa , United States 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿ / ﻿ 43.220361 ° N 93.381417 ° W ﻿ / 43.220361 ; - 93.381417 Coordinates : 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿ / ﻿ 43.220361 ° N 93.381417 ° W ﻿ / 43.220361 ; - 93.381417 Passengers Crew Fatalities 4 ( all ) Survivors 0 Aircraft type Beechcraft Bonanza Operator Dwyer Flying Service , Mason City , Iowa Registration N3794N Flight origin Mason City Municipal Airport , Iowa Destination Hector Airport , North Dakota
  Fact 0.0: The Day the Music Died has a memorial marking the crash site, photographed in 2003.
  Fact 0.1: The accident date was February 3, 1959 (1959-02-03).
  Fact 0.2: The accident summary is spatial disorientation and loss of control in near-IMC conditions.
  Fact 0.3: The accident site is near Clear Lake, Iowa, United States.
  Fact 0.4: The accident site coordinates are 43°13′13.3″N 93°22′53.1″W.
  Fact 0.5: The accident site coordinates are 43.220361°N 93.381417°W.
  Fact 0.6: The accident had 4 fatalities, all of the people involved.
  Fact 0.7: The accident had 0 survivors.
  Fact 0.8: The aircraft type was a Beechcraft Bonanza.
  Fact 0.9: The operator was Dwyer Flying Service, of Mason City, Iowa.
  Fact 0.10: The aircraft registration was N3794N.
  Fact 0.11: The flight origin was Mason City Municipal Airport, Iowa.
  Fact 0.12: The flight destination was Hector Airport, North Dakota.
Sentence 1: At the time , Holly and his band , consisting of Waylon Jennings , Tommy Allsup , and Carl Bunch , were playing on the `` Winter Dance Party '' tour across the United States Midwest . Rising artists Valens and Richardson had joined the tour , as well . The long journeys between venues on board the cold , uncomfortable tour buses adversely affected the performers , with cases of flu and even frostbite . After stopping at Clear Lake to perform , and frustrated by such conditions , Holly chose to charter a plane to reach their next venue in Moorhead , Minnesota . Richardson , who had the flu , swapped places with Jennings , taking his seat on the plane , while Allsup lost his seat to Valens on a coin toss . Soon after takeoff , late at night and in poor , wintry weather conditions , the pilot lost control of the light aircraft , a Beechcraft Bonanza , which subsequently crashed into a cornfield . Everyone on board was killed . The event has since been mentioned in various songs and films . A number of monuments have been erected at the crash site and in Clear Lake , where an annual memorial concert is also held at the Surf Ballroom , the venue that hosted the artists ' last performance .
  Fact 1.0: At the time, Holly and his band were playing on the 'Winter Dance Party' tour across the United States Midwest.
  Fact 1.1: Holly's band consisted of Waylon Jennings, Tommy Allsup, and Carl Bunch.
  Fact 1.2: Rising artists Valens and Richardson had joined the tour.
  Fact 1.3: The long journeys between venues on board the cold, uncomfortable tour buses adversely affected the performers.
  Fact 1.4: The performers had cases of flu.
  Fact 1.5: The performers had cases of frostbite.
  Fact 1.6: After stopping at Clear Lake to perform, and frustrated by such conditions, Holly chose to charter a plane.
  Fact 1.7: Holly chartered the plane to reach the next venue in Moorhead, Minnesota.
  Fact 1.8: Richardson, who had the flu, swapped places with Jennings, taking Jennings's seat on the plane.
  Fact 1.9: Allsup lost his seat to Valens on a coin toss.
  Fact 1.10: Soon after takeoff, late at night and in poor, wintry weather conditions, the pilot lost control of the light aircraft.
  Fact 1.11: The light aircraft was a Beechcraft Bonanza.
  Fact 1.12: The aircraft subsequently crashed into a cornfield.
  Fact 1.13: Everyone on board was killed.
  Fact 1.14: The event has since been mentioned in various songs and films.
  Fact 1.15: A number of monuments have been erected at the crash site.
  Fact 1.16: A number of monuments have been erected in Clear Lake.
  Fact 1.17: An annual memorial concert is also held at the Surf Ballroom.
  Fact 1.18: The Surf Ballroom was the venue that hosted the artists' last performance.

Retrieved items:
<item 1>
The Day the Music Died
The rest of the party would have picked him up in Moorhead , saving him the journey in the bus and leaving him time to get some rest . Flight arrangements (
edit ) A V - tailed Bonanza similar to N3794N , the accident aircraft Surf Ballroom manager Anderson called Hubert Jerry Dwyer ( 1930 - 2016 ) , owner of the
Dwyer Flying Service , a company in Mason City , Iowa , to charter the plane to fly to Hector Airport in Fargo , the closest one to Moorhead . Flight
arrangements were made with Roger Peterson , a 21 - year - old local pilot described as a `` young married man who built his life around flying '' . The flying
service charged a fee of $36 per passenger for the flight on the 1947 single - engined , V - tailed Beechcraft 35 Bonanza ( registration N3794N ) , which could
seat three passengers plus the pilot . A popular misconception , originating from Don McLean 's eponymous song about the crash , was that the plane was called
American Pie . In fact , no record exists of any name ever having been given to N3794N .
</item 1>

<item 2>
The Day the Music Died
The tail light was then observed gradually descending until it disappeared out of view . Around 1 : 00 am , when Peterson failed to make the expected radio
contact , repeated attempts to establish communication were made , at Dwyer 's request , by the radio operator , but they were all unsuccessful . Later that
morning , Dwyer , having heard no word from Peterson since his departure , took off in another airplane to retrace his planned route . Within minutes , at
around 9 : 35 am , he spotted the wreckage less than 6 mi ( 9.7 km ) northwest of the airport . The sheriff 's office , alerted by Dwyer , dispatched Deputy
Bill McGill , who drove to the crash site , a cornfield belonging to Albert Juhl . The Bonanza had impacted terrain at high speed , estimated to have been
around 170 mph ( 270 km / h ) , banked steeply to the right and in a nose - down attitude .
</item 2>

<item 3>
The Day the Music Died
The request was made by L.J. Coon , a retired pilot from New England who felt that the conclusion of the 1959 investigation was inaccurate . Coon suspected a
possible failure of the right rudder , or a problem with the fuel system , as well as a possible improper weight distribution . Coon also argued that Peterson
may have tried to land the plane and that his efforts should be recognized . In April 2015 , the NTSB declined the request , citing that the evidence presented
by Coon was insufficient to merit the reconsideration of the original findings . Legacy ( edit ) Notification of victims ' families ( edit ) Following the
miscarriage suffered by Holly 's wife and the circumstances in which she was informed of his death , a policy was later adopted by authorities not to disclose
victims ' names until after their families have been informed . Memorials ( edit ) A memorial service for Roger Peterson was held at Redeemer Lutheran Church ,
Ventura , Iowa , on February 5 . A funeral was held the next day at St. Paul Lutheran Church in his hometown of Alta ; Peterson was buried in Buena Vista
Memorial Cemetery in nearby Storm Lake .
</item 3>

<item 4>
Passengers of the RMS Titanic
Frank M. Warren , Sr . George Dennick Wick George Dunton Widener Harry Elkins Widener Duane Williams George Henry Wright Roderick Chisholm Survivors ( last
living ) Rhoda Abbott Trevor Allison Lillian Asplund Madeleine Astor Ruth Becker Lawrence Beesley Karl Behr Dickinson Bishop Mauritz Håkan Björnström -
Steffansson Elsie Bowerman Francis Browne Margaret `` Molly '' Brown Daniel Buckley Alden Caldwell Helen Churchill Candee Charlotte Drake Cardeza Lucile Carter
Gladys Cherry Millvina Dean Margaret Devaney Sir Cosmo Duff - Gordon Lucy , Lady Duff - Gordon Ethel Flora Fortune Dorothy Gibson Archibald Gracie IV Frank John
William Goldsmith Edith Haisman Henry S. Harper Eva Hart Margaret Bechstein Hays Masabumi Hosono J. Bruce Ismay Eleanor Ileen Johnson Louise Laroche Louise Kink
Margaret Mannion Michel Marcel Navratil Alfred Nourney Arthur Godfrey Peuchen Winnifred Quick Marjorie Newell Robb Edith Rosenbaum Noël Leslie ,
</item 4>

<item 5>
The Day the Music Died
Richardson , who had the flu , swapped places with Jennings , taking his seat on the plane , while Allsup lost his seat to Valens on a coin toss . Soon after
takeoff , late at night and in poor , wintry weather conditions , the pilot lost control of the light aircraft , a Beechcraft Bonanza , which subsequently
crashed into a cornfield . Everyone on board was killed . The event has since been mentioned in various songs and films . A number of monuments have been
erected at the crash site and in Clear Lake , where an annual memorial concert is also held at the Surf Ballroom , the venue that hosted the artists ' last
performance . Contents ( hide ) 1 Background 2 Flight arrangements 3 Take - off and crash 4 Aftermath 5 Official investigation 6 Subsequent investigations 7
Legacy 7.1 Notification of victims ' families 7.2 Memorials 7.2. 1 Films 7.2. 2 Memorial concerts 7.2. 3 Monuments 7.2. 4 Roads 7.2.
</item 5>

<item 6>
The Day the Music Died
The right wing tip had struck the ground first , sending the aircraft cartwheeling across the frozen field for 540 feet ( 160 m ) , before coming to rest
against a wire fence at the edge of Juhl 's property . The bodies of Holly and Valens had been ejected from the torn fuselage and lay near the wreckage .
Richardson 's body had been thrown over the fence and into the cornfield of Juhl 's neighbor Oscar Moffett , while pilot Peterson 's body was entangled in the
plane 's wreckage . With the rest of the entourage en route to Minnesota , ballroom manager Carroll Anderson , who had driven the party to the airport and
witnessed the plane 's takeoff , had to identify the bodies of the musicians . County coroner Ralph Smiley certified that all four victims died instantly ,
citing the cause of death as `` gross trauma to brain '' for the three artists and `` brain damage '' for the pilot . Aftermath ( edit ) Holly 's pregnant wife
, María Elena , learned of his death from the reports on television .
</item 6>

<item 7>
The Day the Music Died
The musician 's son , Jay Perry , took the opportunity to have his father 's body re-examined to verify the original findings , and asked forensic
anthropologist William Bass to carry out the procedure . Among the rumors surrounding the accident that this second examination sought to verify was that an
accidental firearm discharge took place on board the aircraft and caused the crash , since two months after the event , a farmer had found at the crash site a .
22 ( 5.6 mm ) pistol known to have belonged to Buddy Holly . Another rumor had Richardson surviving the initial impact and crawling out of the aircraft in
search for help , prompted by the fact that his body was found farther from the wreckage than the other three . Bass and his team took several X-rays of
Richardson 's body and eventually concluded that the musician had indeed died instantly from extensive , nonsurvivable fractures to almost all of his bones ; no
traces of lead were found from any bullet . Coroner Smiley 's original report was , therefore , confirmed . In March 2015 , the National Transportation Safety
Board ( NTSB ) received a request to reopen the investigation into the accident . The request was made by L.J.
</item 7>

<item 8>
The Day the Music Died
The Day the Music Died - wikipedia The Day the Music Died Jump to : navigation , search This article is about the plane crash . For other uses , see The Day the
Music Died ( disambiguation ) . The Day the Music Died The memorial marking the crash site , 2003 Accident summary Date February 3 , 1959 ( 1959 - 02 - 03 )
Summary Spatial disorientation , loss of control in near - IMC conditions Site Near Clear Lake , Iowa , United States 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿
/ ﻿ 43.220361 ° N 93.381417 ° W ﻿ / 43.220361 ; - 93.381417 Coordinates : 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿ / ﻿ 43.220361 ° N 93.381417 ° W ﻿ /
43.220361 ; - 93.381417 Passengers Crew Fatalities 4 ( all ) Survivors 0 Aircraft type Beechcraft Bonanza Operator Dwyer Flying Service ,
</item 8>

<item 9>
The Day the Music Died
In fact , no record exists of any name ever having been given to N3794N . The most widely accepted version of events was that Richardson had contracted flu
during the tour and asked Waylon Jennings for his seat on the plane . When Holly learned that Jennings was not going to fly , he said in jest : `` Well , I hope
your ol ' bus freezes up . '' Jennings responded : `` Well , I hope your ol ' plane crashes '' , a humorous but ill - fated response that haunted him for the
rest of his life . Ritchie Valens , who once had a fear of flying , asked Tommy Allsup for his seat on the plane . The two agreed to toss a coin to decide . Bob
Hale , a DJ with KRIB - AM , was working the concert that night and flipped the coin in the ballroom 's side - stage room shortly before the musicians departed
for the airport . Valens won the coin toss for the seat on the flight . In contradiction to the testimony of Allsup and Jennings , Dion has since said that
Holly approached him along with Valens and The Bopper to join the flight , not Holly 's bandmates .
</item 9>

<item 10>
The Day the Music Died
Official investigation ( edit ) The wreckage of N3794N as found by the authorities on the morning of the accident The official investigation was carried out by
the Civil Aeronautics Board ( CAB , precursor to the NTSB ) . It emerged that pilot Roger Peterson had over four years of flying experience , of which one with
Dwyer Flying Service , and had accumulated 711 flying hours , of which 128 on Bonanzas . He had also logged 52 hours of instrument flying training , although
had passed only his written examination , and was not yet qualified to operate in weather that required flying solely by reference to instruments . He and Dwyer
Flying Service itself were certified to operate only under visual flight rules , which essentially require that the pilot must be able to see where he is going
. However , on the night of the accident , visual flight would have been virtually impossible due to the low clouds , the lack of a visible horizon , and the
absence of ground lights over the sparsely populated area .
</item 10>
=== END ITEM id=24f318d3f9dba1e7 ===
