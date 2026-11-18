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

=== ITEM id=e119f09dbe2b53ec ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: Super Bowl LII followed the 2017 season.
  Fact 0.2: The Eagles took their revenge in Super Bowl LII.
  Fact 0.3: The Eagles won Super Bowl LII 41 to 33.

Retrieved items:
<item 1>
Graph query row:
game: Super Bowl LII
date: 2018
</item 1>

<item 2>
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 2>

<item 3>
Super Bowl XXXIX
The next year , the Patriots , having faltered down the stretch , entered those playoffs as the second seed behind the Broncos , who then defeated New England
in the AFC Title Game that year en route to a title of their own in Super Bowl 50 . Two years later , the Patriots , defending champions themselves from Super
Bowl LI , returned to repeat their title only to lose the aforementioned Super Bowl LII . Final statistics ( edit ) Sources : NFL.com Super Bowl XXXIX , Super
Bowl XXXIX Play Finder NE , Super Bowl XXXIX Play Finder Phi Statistical comparison ( edit ) New England Patriots Philadelphia Eagles First downs 21 24 First
downs rushing 6 First downs passing 14 18 First downs penalty Third down efficiency 4 / 12 9 / 16 Fourth down efficiency 0 / 0 0 / 0 Net yards rushing 112 45
Rushing attempts 28 17 Yards per rush 4.0 2.
</item 3>

<item 4>
Super Bowl XXXIX
Coast Guard Academy , and U.S. Army Herald Trumpets . Coin toss Youth football players from Jacksonville : Tyler Callahan , Tyler Deal , Lawrence McCauley , and
Jacob Santana ; New Orleans NFL Junior Player Development coach Tamaris Jackson Halftime show Paul McCartney TV in the United States Network Fox Announcers Joe
Buck , Troy Aikman , Cris Collinsworth , Pam Oliver and Chris Myers Nielsen ratings 41.1 ( est. 86 million viewers ) Market share 62 Cost of 30 - second
commercial $2.3 million ← XXXVIII Super Bowl XL → Super Bowl XXXIX was an American football game played between the American Football Conference ( AFC )
champion New England Patriots and the National Football Conference ( NFC ) champion Philadelphia Eagles to decide the National Football League ( NFL ) champion
for the 2004 season . The Patriots defeated the Eagles by the score of 24 -- 21 . The game was played on February 6 , 2005 , at Alltel Stadium in Jacksonville ,
Florida , the first time the Super Bowl was played in that city .
</item 4>

<item 5>
Super Bowl XXXIX
The Eagles failed to recover their ensuing onside kick attempt , with Christian Fauria catching the ball cleanly and sliding down to seal possession for New
England . The Patriots then played it safe by running the ball 3 times and forcing Philadelphia to use all of its timeouts . New England punter Josh Miller then
pinned the Eagles back at their own 4 - yard line with just 46 seconds left in the game . Philadelphia then tried one last desperate drive to win or tie the
game . But on first down , McNabb was pressured into making a rushed pass to Westbrook at the line of scrimmage . Instead of dropping the pass to stop the clock
, Westbrook made the mistake of catching the ball and was immediately tackled for no gain , keeping the clock running and forcing the Eagles to run back to the
line of scrimmage for their next play with no huddle . On second down , McNabb threw an incomplete pass intended for Owens . Finally on third down , McNabb
threw a pass that deflected off of the outstretched fingertips of Smith and into the arms of Harrison for an interception with nine seconds left .
</item 5>

<item 6>
Philadelphia Eagles
The Eagles then won six consecutive games , including signature wins at the Carolina Panthers , 28 -- 23 , and versus the Washington Redskins , 34 -- 24 , in a
Monday Night contest . On the morning of October 31 , 2017 , just before the NFL trade deadline , the Eagles made a blockbuster move sending a fourth - round
pick to the Miami Dolphins for star running back Jay Ajayi . The move immediately paid dividends for the Eagles heading into their next game versus the Denver
Broncos as Ajayi rushed for 77 yards on just eight attempts including a 46 - yard TD near the end of the second quarter . Ajayi and Wentz led the Eagles to a 51
-- 23 rout of the top defense in the NFL . Foles in the Huddle before leaving the field before Super Bowl LII . The Eagles headed into their bye week with the
best record in the NFL ( 8 -- 1 ) , and with Wentz leading the NFL MVP discussion with an NFL - leading 23 touchdowns to 5 interceptions .
</item 6>

<item 7>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 7>

<item 8>
Philadelphia Eagles
Philadelphia Eagles . September 9 , 2016 . Retrieved September 17 , 2017 . Jump up ^ `` Philadelphia Eagles Team Capsule '' ( PDF ) . 2017 Official National
Football League Record and Fact Book . National Football League . August 22 , 2017 . Retrieved September 17 , 2017 . Jump up ^ `` Top 10 NFL Rivalries Of All
Time : No. 4 Giants - Eagles '' . Sports Illustrated . December 15 , 2005 . Retrieved September 17 , 2017 . Jump up ^ Chadiha , Jeffri ( October 31 , 2007 ) .
`` Ranking the NFL 's best rivalries : Where does Colts - Pats fit ? '' . ESPN.com . Retrieved April 12 , 2008 . Jump up ^ Bryan , Dave ( September 20 , 2016 )
. `` After 8 Straight Losses , Steelers Looking For Philadelphia Freedom Sunday Against Eagles '' . SteelersDepot.com . Retrieved September 17 , 2017 .
</item 8>

<item 9>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 9>

<item 10>
Philadelphia Eagles
During their bye , the Eagles signed former Giants offensive tackle Will Beatty and former Saints linebacker Dannell Ellerbe to not only add depth but also
players with Super Bowl experiences . Coming off of the bye week , the Philadelphia Eagles entered a Sunday night showdown with the defending NFC East champion
Dallas Cowboys . The Eagles came out sluggish and trailed 9 -- 7 at the half . The team exploded in the second half though , rattling off 30 unanswered points
to soundly defeat their rival by a final score of 37 -- 9 . The Eagles were rolling on a nine - game winning streak after a huge blowout win over the Chicago
Bears . The Eagles traveled to Seattle in Week 13 in a huge Sunday Night showdown against Russell Wilson and the Seattle Seahawks . The Eagles struggled
throughout the whole game , blowing many scoring opportunities and allowing Wilson to throw 3 touchdowns . The Eagles lost 24 -- 10 , ending their win streak .
However , the Eagles bounced back in Week 14 win over the Los Angeles Rams in a tough 43 -- 35 win . However , Carson Wentz left the game in the third quarter
with a knee injury .
</item 10>
=== END ITEM id=e119f09dbe2b53ec ===

=== ITEM id=02eb0d55b06dfe83 ===
Gold sentences and facts:
Sentence 0: A transfer case is a part of the drivetrain of four - wheel - drive , all - wheel - drive , and other multiple powered axle vehicles . The transfer case transfers power from the transmission to the front and rear axles by means of drive shafts . It also synchronizes the difference between the rotation of the front and rear wheels , and may contain one or more sets of low range gears for off - road use .
  Fact 0.0: A transfer case is a part of the drivetrain of four-wheel-drive, all-wheel-drive, and other multiple powered axle vehicles.
  Fact 0.1: A transfer case transfers power from the transmission to the front and rear axles by means of drive shafts.
  Fact 0.2: A transfer case synchronizes the difference between the rotation of the front and rear wheels.
  Fact 0.3: A transfer case may contain one or more sets of low range gears for off-road use.

Retrieved items:
<item 1>
Transfer case
See also ( edit ) NP - 205 AMC / Jeep Transmissions Jeep four wheel drive systems Mitsubishi Super Select References ( edit ) Jump up ^
http://www.cdxetextbook.com/trans/finalDrives/allWheel/transfercasediff.html Transfer case differential action Jump up ^
http://www.fourwheeler.com/techarticles/68958_how_transfer_cases_work/index.html How Transfer Cases Work Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Transfer_case&oldid=805616544 '' Categories : Vehicle technology Four - wheel drive layout Hidden categories : All
articles with unsourced statements Articles with unsourced statements from May 2015 Articles with unsourced statements from March 2012 Talk Contents About
Wikipedia فارسی 日本 語 Polski Русский Simple English Svenska Українська Edit links This page was last edited on 16 October 2017 , at 15 : 12 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy
.
</item 1>

<item 2>
Transfer case
Those used in sports cars and performance sedans are usually `` transparent '' to the driver ; there is no shifter or select lever . Drive type ( edit ) Gear -
driven ( edit ) There are two different types of internal power - transfer mechanism found in most transfer cases . Gear - driven transfer cases use sets of
gears to drive either the front or both the front and rear driveshafts . These are generally strong , heavy units that are used in large trucks , but there are
currently several gear drive cases in production for passenger cars . Chain - driven ( edit ) Chain - driven transfer cases use a chain to drive most often only
one axle , but can drive both axles . Chain - driven transfer cases are quieter and lighter than gear - driven ones . They are used in vehicles such as compact
trucks , full size trucks , Jeeps and SUVs . Some off - road driving enthusiasts modify their vehicles to use gear - driven transfer cases , accepting the
additional weight and noise to gain the extra strength they generally provide . Housing type ( edit ) Married ( edit ) Transfer cases are also classified as
either `` divorced '' / independent or `` married '' .
</item 2>

<item 3>
Transfer case
1 M.S.O.F. 2.3. 2 E.S.O.F. 3 See also 4 References Functions ( edit ) The transfer case receives power from the transmission and sends it to both the front and
rear axles . This can be done with gears , hydraulics , or chain drive . On some vehicles , such as four - wheel - drive trucks or vehicles intended for off -
road use , this feature is controlled by the driver . The driver can put the transfer case into either `` two - wheel - drive '' or `` four - wheel - drive ''
mode . This is sometimes accomplished by means of a shifter , similar to that in a manual transmission . On some vehicles this may be electronically operated by
a switch instead . Some vehicles , such as all - wheel - drive sports cars , have transfer cases that are not selectable . Such a transfer case is permanently
`` locked '' into all - wheel - drive mode . Transfer cases that are designed to allow for normal road use synchronize the difference between the rotation of
the front and rear wheels , in much the same way the differential acts on a given axle . This is necessary because the front and rear tires never turn at the
same speed .
</item 3>

<item 4>
Transfer case
Married transfer cases are bolted directly to the transmission , usually between the transmission 's output shaft and the rear or main driveshaft . Sometimes a
married transfer case is an integral part of the transmission and the two components share the same housing or `` case '' , as is commonly found on recent
Subaru products and some other all - wheel - drive cars . Divorced / independent ( edit ) A divorced or independent transfer case is completely separate from
the transmission . It is located further down the driveline than a married transfer case and connected to the transmission output shaft by a short driveshaft .
Independent transfer cases are used on very long wheelbase vehicles , such as commercial trucks or military trucks . Transfer case Shift type ( edit ) M.S.O.F.
( edit ) Manual Shift On - the - Fly transfer cases have a selector lever on the driver 's side floor transmission hump and may also have either two sealed
automatic front axle locking hubs or two manual front axle hub selectors of `` LOCK '' and `` UNLOCK '' or `` FREE '' . To engage the four - wheel - drive
system the vehicle must be moving at a low speed , the speed at which 4x4 can be engaged depends on the vehicle .
</item 4>

<item 5>
Transfer case
This is necessary because the front and rear tires never turn at the same speed . Different rates of tire rotation are generally due to different tire diameters
( since front and rear tires inevitably wear at different rates ) and different gear ratios in the front and rear differentials since manufacturers will often
have a slightly lower ratio in the front vs. the rear to help with control ( such as a 3.55 : 1 in the rear differential and a 3.54 : 1 in the front
differential ) . If the transfer case did not make up the difference between the two different rates of rotation , binding would occur and the transfer case
could become damaged . This is also why a transfer case that is not designed for on - road use must never be driven on dry pavement . Transfer cases designed
for off - road use can mechanically lock the front and rear axles when needed ( e.g. when one of the axles is on a slippery surfaces or stuck in mud , whereas
the other has better traction ) . This is the equivalent to the differential lock . The transfer case may contain one or more sets of low range gears for off -
road use .
</item 5>

<item 6>
Transfer case
This is only for the four - wheel - drive high setting . To engage the four - wheel - drive low setting , the vehicle must be stopped and the transmission must
be shifted to neutral , then the four - wheel - drive low can be selected . ESOF ( edit ) Electronic Shift On - the - Fly ( ESOF ) transfer cases have a dash -
mounted selector switch or buttons with front sealed automatic locking axle hubs or drive flanges . Some vehicles , such as Jeeps , also have what is called
SelecTrac , which has a slider switch on the center console . Unlike the manual transfer case , this system has a transfer case motor . To engage the four -
wheel - drive system the vehicle must be moving at a lower speeds , the speed at which 4x4 can be engaged depends on the vehicle . This is only for the four -
wheel - drive high setting . To engage the four - wheel - drive low setting , the vehicle must be stopped and the transmission must be shifted to neutral , then
the four - wheel - drive low can be selected .
</item 6>

<item 7>
Jeep
Off - road abilities Jeep Wrangler off - roading . Jeep advertising has always emphasized the brand 's vehicles ' off - road capabilities . Today , the Wrangler
is one of the few remaining four - wheel - drive vehicles with solid front and rear axles . These axles are known for their durability , strength , and
articulation . New Wranglers come with a Dana 44 rear differential and a Dana 30 front differential . The upgraded Rubicon model of the JK Wrangler is equipped
with electronically activated locking differentials , Dana 44 axles front and rear with 4.10 gears , a 4 : 1 transfer case , electronic sway bar disconnect and
heavy duty suspension . Another benefit of solid axle vehicles is they tend to be easier and cheaper to `` lift '' with aftermarket suspension systems . This
increases the distance between the axle and chassis of the vehicle . By increasing this distance , larger tires can be installed , which will increase the
ground clearance , allowing it to traverse even larger and more difficult obstacles . In addition to higher ground clearance , many owners aim to increase
suspension articulation or `` flex '' to give their Jeeps greatly improved off - road capabilities .
</item 7>

<item 8>
Transfer case
Transfer case - wikipedia Transfer case Jump to : navigation , search For the US military use of the term , see coffin . Inside of a 231 New Process Gear
transfer case . Part - time / Manual , shift on the fly A transfer case is a part of the drivetrain of four - wheel - drive , all - wheel - drive , and other
multiple powered axle vehicles . The transfer case transfers power from the transmission to the front and rear axles by means of drive shafts . It also
synchronizes the difference between the rotation of the front and rear wheels , and may contain one or more sets of low range gears for off - road use .
Contents ( hide ) 1 Functions 2 Types 2.1 Drive type 2.1. 1 Gear - driven 2.1. 2 Chain - driven 2.2 Housing type 2.2. 1 Married 2.2. 2 Divorced / independent
2.3 Transfer case shift type 2.3. 1 M.S.O.F. 2.3. 2 E.S.O.F.
</item 8>

<item 9>
Transfer case
The transfer case may contain one or more sets of low range gears for off - road use . Low range gears are engaged with a shifter or electronic switch . On many
transfer cases , this shifter is the same as the one that selects 2WD or 4WD operation . Low range gears allow the vehicle to drive at much slower speeds while
still operating within the usable power band / RPM range of the engine . This also increases the torque available at the axles . Low - range gears are used
during slow - speed or extreme off - road maneuvers , such as rockcrawling , navigating dangerous roads , or when pulling a heavy load . This feature is often
absent on all - wheel - drive cars . Some very large vehicles , such as heavy equipment or military trucks , have more than one low - range gear . Types ( edit
) Transfer cases used on `` part - time '' four - wheel - drive off - road vehicles such as trucks , truggies , rock - crawling vehicles , and some military
vehicles generally allow the driver to select 2WD or 4WD , as well as high or low gear ranges .
</item 9>

<item 10>
Transfer case
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 10>
=== END ITEM id=02eb0d55b06dfe83 ===

=== ITEM id=0d02dd3fb5aed71d ===
Gold sentences and facts:
Sentence 0: In the 12th century , luxury increased and bedsteads were made of wood much decorated with inlaid , carved , and painted ornamentation . They also used folding beds , which served as couches by day and had cushions covered with silk laid upon leather . At night a linen sheet was spread and pillows placed , while silk - covered skins served as coverlets . The Carolingian manuscripts show metal bedsteads much higher at the head than at the feet , and this shape continued in use until the 13th century in France , many cushions being added to raise the body to a sloping position . In 12th - century manuscripts , the bedsteads appear much richer , with inlays , carving , and painting , and with embroidered coverlets and mattresses in harmony . Curtains were hung above the bed and a small hanging lamp is often shown .
  Fact 0.0: In the 12th century, luxury increased.
  Fact 0.1: In the 12th century, bedsteads were made of wood.
  Fact 0.2: The wooden bedsteads were much decorated with inlaid, carved, and painted ornamentation.
  Fact 0.3: Folding beds were used.
  Fact 0.4: The folding beds served as couches by day.
  Fact 0.5: The folding beds had cushions covered with silk laid upon leather.
  Fact 0.6: At night a linen sheet was spread on the bed.
  Fact 0.7: At night pillows were placed on the bed.
  Fact 0.8: At night silk-covered skins served as coverlets.
  Fact 0.9: Carolingian manuscripts show metal bedsteads much higher at the head than at the feet.
  Fact 0.10: The shape of the metal bedsteads with a higher head continued in use until the 13th century in France.
  Fact 0.11: Many cushions were added to raise the body to a sloping position.
  Fact 0.12: In 12th-century manuscripts, bedsteads appear much richer.
  Fact 0.13: The richer 12th-century bedsteads have inlays, carving, and painting.
  Fact 0.14: The richer 12th-century bedsteads have embroidered coverlets and mattresses in harmony.
  Fact 0.15: Curtains were hung above the bed.
  Fact 0.16: A small hanging lamp is often shown.

Retrieved items:
<item 1>
Bed
The mattresses were often filled with pea - shucks , straw , or feathers . At this time great personages were in the habit of carrying most of their property
about with them , including beds and bed - hangings , and for this reason the bedsteads were for the most part mere frameworks to be covered up ; but about the
beginning of the 16th century bedsteads were made lighter and more decorative , since the lords remained in the same place for longer periods . Renaissance and
modern Europe In the 17th century , which has been called `` the century of magnificent beds '' , the style a la duchesse , with tester and curtains only at the
head , replaced the more enclosed beds in France , though they lasted much longer in England . Louis XIV had an enormous number of sumptuous beds , as many as
413 being described in the inventories of his palaces . Some of them had embroideries enriched with pearls , and figures on a silver or golden ground . The
great bed at Versailles had crimson velvet curtains on which `` The Triumph of Venus '' was embroidered . So much gold was used that the velvet scarcely showed
.
</item 1>

<item 2>
Bed
During the 17th century this curious custom became general , perhaps to avoid the tiresome details of etiquette . Portable beds were used in high society in
France till the end of the Ancien Régime . The earliest of which mention has been found belonged to Charles the Bold . They had curtains over a light framework
, and were in their way as fine as the stationary beds . Iron beds appear in the 18th century ; the advertisements declare them as free from the insects which
sometimes infested wooden bedsteads . Elsewhere , there was also the closed bed with sliding or folding shutters , and in England -- where beds were commonly
quite simple in form -- the four poster was the usual citizen 's bed until the middle of the 19th century . Bed sizes Main article : Bed size Bed sizes vary
considerably around the world , with most countries having their own standards and terminology . While the `` double '' size appears to be standard among
English speaking countries , based on the imperial measurement of 4 ft 6 in by 6 ft 3 in ( 137 cm x 190 cm ) , the sizes for other bed types tend to vary .
</item 2>

<item 3>
In the 14th century, beds were adorned with silk, velvet, fur, and embroidery; ceiling- or wall-supported four-poster beds evolved into curtained
rooms-within-rooms. The space beside the bed, the ruelle, hosted intimate visitors, and feather mattresses became prized. By the 15th century, beds reached
roughly 7–8 by 6–7 feet, with straw, pea-shuck, or feather fillings. Notable examples include the Great Bed of
</item 3>

<item 4>
Bed
Contents ( hide ) 1 History 1.1 Ancient world 1.2 Medieval Europe 1.3 Renaissance and modern Europe 2 Bed sizes 3 Notable examples 4 Types 5 Frames 6 See also 7
References 8 External links History Ancient world Tutankhamun 's gilded bed from the 14th century BC Early beds were little more than piles of straw or some
other natural material ( e.g. a heap of palm leaves , animal skins , or dried bracken ) . An important change was raising them off the ground , to avoid drafts
, dirt , and pests . Bedding dated around to 3600 BC was discovered in Sibudu Cave , South Africa . The bedding consists of sedge and other monocotyledons
topped with the leaves of Cryptocarya woodii Engl . Beds found in a preserved northern Scottish village , which were raised boxes made of stone and likely
topped with comfortable fillers , were dated to between 3200 BC and 2200 BC . The Egyptians had high bedsteads which were ascended by steps , with bolsters or
pillows , and curtains to hang around .
</item 4>

<item 5>
Bed
So much gold was used that the velvet scarcely showed . Napoleon I 's bed In the 18th century feather pillows were first used as coverings in Germany , which in
the fashions of the bed and the curious etiquette connected with the bedchamber followed France for the most part . The beds were a la duchesse , but in France
itself there was great variety both of name and shape . The custom of the `` bed of justice '' upon which the king of France reclined when he was present in
parliament , the princes being seated , the great officials standing , and the lesser officials kneeling , was held to denote the royal power even more than the
throne . Louis XI is credited with its first use and the custom lasted till the end of the monarchy . In the chambre de parade , where the ceremonial bed was
placed , certain persons , such as ambassadors or great lords , whom it was desired to honour , were received in a more intimate fashion than the crowd of
courtiers . At Versailles women received their friends in their beds , both before and after childbirth , during periods of mourning , and even directly after
marriage -- in fact in any circumstances which were thought deserving of congratulation or condolence .
</item 5>

<item 6>
Portable beds became common in 17th-century France, likely simplifying etiquette, and remained fashionable until the Ancien Régime ended. Charles the Bold owned
the earliest recorded example; such beds had curtains on light frames and could be as elaborate as fixed beds. Iron beds appeared in the 18th century,
advertised as insect-resistant. Other designs included shuttered beds; four-posters were common in England until the mid-19th century. Bed sizes vary worldwide;
the English-speaking “double
</item 6>

<item 7>
Bed
Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( April 2015 ) ( Learn how and
when to remove this template message ) See also : Category : Beds Lit à la Polonaise ( Polish style bed ) , Royal Castle in Warsaw , 18th century . Patent #
322,177 , on 14 July 1885 issued to Sarah E. Goode for a cabinet bed Drawing of a candle - lit mourning bed ( Trauergerüst ) for abbess Franziska Christine von
Pfalz - Sulzbach , 1776 Chinese style beds There are many varieties of beds : An adjustable bed is a bed that can be adjusted to a number of different positions
. Most hospital beds are adjustable , so that a patient can have different parts of her body elevated for medical reasons . Some people have adjustable beds in
private homes . Some adjustable beds are designed for couples ; they use two separate mattresses and adjustment mechanisms . This permits one partner to be
lying flat to sleep while the other has the head and shoulders elevated to watch television or read .
</item 7>

<item 8>
Early beds were piles of straw, leaves, skins, or other natural materials; raising them off the ground helped avoid drafts, dirt, and pests. Bedding from about
3600 BC was found in South Africa, and raised stone beds from 3200–2200 BC in Scotland. Egyptians used elevated beds with steps, pillows, and curtains. Romans
had distinct beds for sleeping, marriage, dining, studying, and carrying the dead. Medieval Europeans used floor bedding or benches with mattresses stuffed with
</item 8>

<item 9>
Bed
Silk , velvet , and even cloth of gold were frequently used . Inventories from the beginning of the 14th century give details of these hangings lined with fur
and richly embroidered . It was then that the Four poster bed ( also known as a tester bed ) made its first appearance , the bed being slung from the ceiling or
fastened to the walls , a form which developed later into a room within a room , shut in by double curtains , sometimes even to exclude all drafts . The space
between bed and wall was called the ruelle , and very intimate friends were received there . The 14th century is also the time when feather beds became highly
prized possessions . In the 15th century beds became very large , reaching 7 to 8 feet ( 2.1 to 2.4 m ) by 6 to 7 feet ( 1.8 to 2.1 m ) . The mattresses were
often filled with pea - shucks , straw , or feathers .
</item 9>

<item 10>
From the 12th century, beds grew more luxurious, with carved, painted or inlaid frames, silk coverings, embroidered linens, curtains and lamps. Mattresses used
straw, pea-shucks or feathers; portable beds were often simple frameworks. By the 14th century, rich hangings overshadowed the woodwork, while 16th-century beds
became lighter and more decorative. The 17th century prized magnificent beds, including Louis XIV’s lavish examples. In the 18
</item 10>
=== END ITEM id=0d02dd3fb5aed71d ===

=== ITEM id=c2aded53e1a0a0b0 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Italian Title / RAI English Title 4Kids English title Italian air date American air date `` Una Fata a Gardenia '' / `` An Unexpected Event '' `` It Feels Like Magic '' 28 January 2004 ( 2004 - 01 - 28 ) 19 June 2004 While on summer vacation , Bloom , an ordinary 16 - year - old high school girl living on Earth , encounters a battle between a 17 - year - old fairy named Stella and an ogre named Knut . As Stella 's magic scepter is stolen , Bloom intervenes , discovering that she has magical powers and manages to turn the tide . Stella invites Bloom to attend Alfea , a school for fairies , to fully master her new powers . Knut returns with more ghouls and a troll , launching a surprise attack at Bloom 's home . Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky , Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School '' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents , she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy , Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her . Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school , Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded . She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball . Later Bloom spots the Trix trying to steal Stella 's ring , and tries to fight them off , transforming in a fairy for the first time . Although the Trix are able to escape , the other Winx girls inform Bloom that Stella 's ring is safe and that they swapped the contents of the case with a baby duck ( Pepe ) that bonds with Icy . `` La palude di Melmamora '' / `` The Black - Mud Swamp '' `` The Voice of Nature '' 4 February 2004 ( 2004 - 02 - 04 ) 10 July 2004 As part of Professor Palladium 's field trip , the class must navigate the Black - Mud Swamp to reach a clearing in the Gloomy - Wood Forest by listening to nature and without using any magic . The boys from Red Fountain are transporting a troll prisoner but the Trix cause their aircraft to crash land , and the troll is freed . The Winx and the Red Fountain boys end up meeting in the swamp , and must work together , despite their differences , to find their way out . They encounter the troll who is harassing some other Alfea fairies , and chase him off , however , the Trix find the troll and send him away . Although the Red Fountain boys failed to deliver the troll , they remark they are starting to feel like a team . The Winx girls , although they finished last in their assignment , are commended by their teachers for their helping others . 5 5 `` Appuntamento al buio '' / `` Date with Disaster '' `` Date with Disaster '' 6 February 2004 ( 2004 - 02 - 06 ) 17 July 2004 While the Winx are on kitchen duty , Stella is invited on a date with Prince Sky ( Brandon ) at the brand - new Black Lagoon restaurant and also asks for help on picking out an outfit . While selecting her attire , Stella decides not to wear the ring of Solaria and gives it to Bloom for safekeeping ( believing she wo n't need her magic powers ) . The next morning , Stella starts acting completely different , becoming irritable and refusing to say antything about last night . During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria ! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower '' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out what made the Trix go to such lengths to get it . Meanwhile , the Trix use a spell on the ring , hoping to access an extremely powerful magic known as the Dragon Flame , only to have their spell blow up in their faces as the ring proves negative ! An infuriated Icy throws the ring away and plans to begin the search all over again . Back at Alfea , Bloom learns more on the ring of Solaria , but also learns about the Dragon Flame . Bloom then turns to Faragonda for answers , who tells her that the Dragon Flame was the energy of the Great Dragon , who created the magical universe and later rested on the planet Domino . That night , the Winx plan to infiltrate Cloud Tower to retrieve Stella 's ring ( since she ca n't transform without it ) . Using the same secret tunnel the Trix used earlier , the girls sneak into Cloudtower without a hitch while the Trix are out . After searching their room , Stella finds her ring , but the tunnel does n't reopen and the Winx are forced to find the main exit without getting caught . While exploring , the Winx stumble upon Cloudtower 's archive which contains history on every person in the Magic dimension . A curious Bloom spots a book with her name on it and tries to read more on her past , but quickly regrets it when it turns into a trap that alerts Headmistress Griffin ! The Winx try to make a hasty escape , but Griffin sends Bako monsters to scare them out ! After using their magic to fight the monsters , the Winx are led to a dead end . In a panic to destroy the last monster , Stella accidentally sets the room on fire ! As the Winx begin to suffocate , a mysterious voice guides Bloom to an exit , allowing them to leave Cloudtower . Though the ring was recovered , the Winx are busted since Griffin already called Faragonda hours ago . As punishment , Faragonda temporarily extracts all their powers . 7 7 `` A che servono gli amici ? '' / `` Friends in Need '' `` Grounded '' 11 February 2004 ( 2004 - 02 - 11 ) 31 July 2004 The Winx have lost all their magic as punishment for sneaking into Cloudtower , but to make matters worse , Griselda tasks them with cleaning the entire school without their magic ( or technology , much to Tecna 's dismay ) while the rest of the students and faculty go to a concert . Problems continue as Tecna struggles with no gadgets to help her , causing her to wear a bucket on her head and attempt to use a broom , while Musa , Stella and Bloom have a water fight , wasting their time , energy and supplies . In order to make the punishment lighter , the girls secretly invite the boys from Redfountain over after everyone has left , claiming to be throwing an all - night party . Meanwhile , the Trix use the opportunity to sneak into a near - empty Alfea to find the real source of the Dragon Flame . The boys arrive and instead learn they 'll be working as janitors , though they still play music as they work . Soon after , the Trix teleport to Alfea and summon the vacuums to guide them towards the Dragon Flame , while distracting the gang by summoning a monster . The Winx and specialists quickly meet their distraction , and while the boys try to fight with their weapons , their quickly knocked aside . The Winx , despite having no magic , manage to outsmart the beast by using floor cleaner . Unable to find the Dragon Flame , the Trix depart to Cloudtower , unseen by anyone while the Specialists head back to Redfountain . The next morning , Faragonda sees the damage and hears the story from the Winx and as a reward for defeating an adversary using only wits and resourcefulness , the Winx 's powers are returned to them . 8 8 `` La festa della rosa '' / `` A Friendship Sundered '' `` The Day of the Rose '' 13 February 2004 ( 2004 - 02 - 13 ) 7 August 2004 During the Rose Festival , both Bloom and Riven unwittingly fall into a trap by the Trix to divide the close relationship between the Specialists and the Winx . In the end , Riven thinks that Darcy has saved him in a racing crash and he blames Bloom for what happened then gets mad at his friends for believing Bloom over him and transfers out of the squad of Specialists along with Darcy . 9 9 `` Il tradimento di Riven '' / `` Betrayed ! '' `` Spelled '' 16 February 2004 ( 2004 - 02 - 16 ) 14 August 2004 There are hostilities between the Winx due to Stella 's behavior , and Musa 's feelings for Riven are strained . In a fight with the Trix , the possessor of the Dragon 's Flame is revealed to be Bloom . 10 10 `` La fiamma del drago '' / `` Bloom Tested '' `` Magical Reality Check '' 18 February 2004 ( 2004 - 02 - 18 ) 21 August 2004 It 's time for Bloom 's magical midterm test in a virtual reality machine . The Trix mess with the computer so they can enter it and test her to be sure of Bloom is indeed the one who possesses the Dragon Flame . 11 11 `` Il regno delle ninfe '' / `` The Monster and The Willow '' `` Junior League '' 20 February 2004 ( 2004 - 02 - 20 ) 28 August 2004 Flora experiments with her plants to study for her test . Stella ( who had her clothes torn apart by a plant ) and Tecna ( who was attacked by a plant ) tell Flora to move the plants . She moves them to Black - Mud Swamp . Flora seeks help from the Water Nymphs in finding a special plant . The Water Nymphs tell the Winx about a monster which has been terrorizing them and the Winx decide to fight it . 12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom , which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13 13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop . A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 - 02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March 2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03 ) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17 `` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! '' / `` The Font of Dragon Fire '' `` Senior Witches Go to Earth '' 8 March 2004 ( 2004 - 03 - 08 ) 16 October 2004 Upset about the events at Redfountain , Bloom returns to Earth . The Trix take advantage of the situation and attack her on Earth . The witches reveal to Bloom that she is the surviving heir to planet Domino ( Sparks ) , and they steal the Dragon Fire from her . 19 19 `` Attacco ad Alfea '' / `` The Fall of Magix '' `` The Army of Decay '' 10 March 2004 ( 2004 - 03 - 10 ) 23 October 2004 Bloom returns to Alfea following the events on Earth . The Trix take control of Cloud Tower and summon the Army of Decay to take control of Magix . 20 20 `` La scomparsa di Bloom '' / `` Mission to Domino '' `` Sparks of Hope '' 12 March 2004 ( 2004 - 03 - 12 ) 30 October 2004 To counter the powerful magic possessed by the Trix , the Winx Club go to planet Domino hoping to find the Dragon 's Flame . Redfountain is attacked by the Army of Decay . 21 21 `` Trappola di ghiaccio '' / `` The Crown of Dreams '' `` The Frozen Palace '' 15 March 2004 ( 2004 - 03 - 15 ) 6 November 2004 In the ruins of Domino ( Sparks ) , the Winx are attacked during their exploration . In Magix , Redfountain is destroyed by the Trix . 22 22 `` Il ritorno di Riven '' / `` Storming Cloudtower '' `` Mission to Cloud Tower '' 17 March 2004 ( 2004 - 03 - 17 ) 13 November 2004 The Army of Decay forces the retreat of the population to Alfea . The Winx go to Cloud Tower in attempt to retrieve the power of the Dragon 's Flame from the Trix . Riven rejoins the Specialists . 23 23 `` Fuga da Torrenuvola '' / `` Power Play '' `` The Search for the Flame '' 19 March 2004 ( 2004 - 03 - 19 ) 20 November 2004 The Trix torment the Winx for being unable to find the power of the Dragon 's Flame on Domino ( Sparks ) . The cadre and staff of Cloud Tower escape to Alfea . 24 24 `` Il mistero del lago '' / `` The Witches ' Siege '' `` Battle for Alfea '' 22 March 2004 ( 2004 - 03 - 22 ) 27 November 2004 The Army of Darkness launches an attack on Alfea . Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice . 25 25 `` Il sonno di Magix '' / `` The Ultimate Challenge '' `` The Great Witch Invasion '' 24 March 2004 ( 2004 - 03 - 24 ) 4 December 2004 The first attack on Alfea fails . Bloom follows the spirit of her older - sister Daphne and regains her Dragon Fire powers . The Trix lead the Army of Decay in a final attack on Alfea . 26 26 `` Battaglia finale '' / `` The Witches ' Downfall '' `` Fire and Ice '' 26 March 2004 ( 2004 - 03 - 26 ) 11 December 2004 As the Army of Darkness approaching Alfea and the Trix seeking to take complete control of the Magix Dimension , the Winx , except Bloom and including Alfea 's students and teachers take action to fend off the Trix from completely destroying algae to the ground before the Trix use the powers of the Dragon . RC1 RC1 `` Attack of the Senior Witches '' N / A 22 May 2004 A clip show episode made by 4Kids Entertainment featuring scenes from episodes 18 , 19 and 20 where the Trix attack Alfea , serving as a sneak preview to the first season .
  Fact 0.0: The table lists episodes with columns for the overall number, the season number, the Italian title / RAI English title, the 4Kids English title, the Italian air date and the American air date.
  Fact 0.1: Episode 5, 'Appuntamento al buio', has the RAI English title and the 4Kids English title 'Date with Disaster'.
  Fact 0.2: Episode 5 aired in Italy on 6 February 2004 and in America on 17 July 2004.
  Fact 0.3: While the Winx are on kitchen duty, Stella is invited on a date with Prince Sky (Brandon) at the brand-new Black Lagoon restaurant.
  Fact 0.4: Stella also asks for help picking out an outfit.
  Fact 0.5: While selecting her attire, Stella decides not to wear the ring of Solaria and gives it to Bloom for safekeeping, believing she won't need her magic powers.
  Fact 0.6: The next morning, Stella starts acting completely differently, becoming irritable and refusing to say anything about the previous night.
  Fact 0.7: During classes, Bloom notices that Stella is absent and goes back to their dorm to check.
  Fact 0.8: Bloom finds that Stella has torn the room apart.
  Fact 0.9: An enraged Stella asks Bloom where her ring is, and eventually blasts her when she does not say.
  Fact 0.10: Stella then suddenly vanishes, leaving the other four to find Prince Sky at Redfountain for information.
  Fact 0.11: Prince Sky says he worked out all night and never wrote the invitation.
  Fact 0.12: The Winx decide to find the Black Lagoon.
  Fact 0.13: The Black Lagoon does not show up on Tecna's map.
  Fact 0.14: No one in the city seems to know where the Black Lagoon is.
  Fact 0.15: Just as Musa is ready to quit, a strange man gives the girls the exact location of the Black Lagoon.
  Fact 0.16: The girls find a beautiful restaurant on the edge of town.
  Fact 0.17: Flora finds it strange how the gorgeous tree feels dead.
  Fact 0.18: An illusion has been cast, revealing a run-down shack.
  Fact 0.19: The Winx find Stella inside the shack, and she begins attacking them.
  Fact 0.20: The girls transform and learn that the entire date was a plot by the Trix.
  Fact 0.21: The Trix sent the fake invitation and kidnapped Stella to get her ring of Solaria.
  Fact 0.22: Because Stella did not wear the ring, the Trix sent Darcy disguised as Stella back to Alfea to find it.
  Fact 0.23: The Trix reveal the real Stella trapped in magic energy rings.
  Fact 0.24: The Trix force Bloom into an ultimatum: hand over the ring or Icy will kill Stella.
  Fact 0.25: Bloom decides Stella is more important, transforms out of fairy form and gives the Trix the ring.
  Fact 0.26: Bloom begins her role as the unofficial leader.
  Fact 0.27: Episode 6, 'Missione a Torrenuvola', has the RAI English title 'Mission at Cloudtower' and the 4Kids English title 'Secret Guardian'.
  Fact 0.28: Episode 6 aired in Italy on 9 February 2004 and in America on 24 July 2004.
  Fact 0.29: After returning to Alfea, Bloom researches the Ring of Solaria, hoping to find out what made the Trix go to such lengths to get it.
  Fact 0.30: The Trix use a spell on the ring, hoping to access an extremely powerful magic known as the Dragon Flame.
  Fact 0.31: The Trix's spell blows up in their faces as the ring proves negative.
  Fact 0.32: An infuriated Icy throws the ring away and plans to begin the search all over again.
  Fact 0.33: Bloom learns more about the Ring of Solaria at Alfea.
  Fact 0.34: Bloom learns about the Dragon Flame at Alfea.
  Fact 0.35: Bloom turns to Faragonda for answers.
  Fact 0.36: Faragonda tells Bloom that the Dragon Flame was the energy of the Great Dragon.
  Fact 0.37: The Great Dragon created the magical universe.
  Fact 0.38: The Great Dragon later rested on the planet Domino.
  Fact 0.39: That night, the Winx plan to infiltrate Cloud Tower to retrieve Stella's ring, since she cannot transform without it.
  Fact 0.40: The Winx use the same secret tunnel the Trix used earlier.
  Fact 0.41: The Winx sneak into Cloudtower without a hitch while the Trix are out.
  Fact 0.42: After searching their room, Stella finds her ring.
  Fact 0.43: The tunnel does not reopen, so the Winx must find the main exit without getting caught.
  Fact 0.44: The Winx stumble upon Cloudtower's archive, which contains history on every person in the Magic dimension.
  Fact 0.45: Bloom spots a book with her name on it and tries to read more about her past.
  Fact 0.46: The book turns into a trap that alerts Headmistress Griffin.
  Fact 0.47: Griffin sends Bako monsters to scare the Winx out.
  Fact 0.48: After using their magic to fight the monsters, the Winx are led to a dead end.
  Fact 0.49: In a panic to destroy the last monster, Stella accidentally sets the room on fire.
  Fact 0.50: As the Winx begin to suffocate, a mysterious voice guides Bloom to an exit, allowing the Winx to leave Cloudtower.
  Fact 0.51: The Winx's ring was recovered.
  Fact 0.52: The Winx are busted since Griffin already called Faragonda hours ago.
  Fact 0.53: As punishment, Faragonda temporarily extracts all the Winx's powers.
  Fact 0.54: Episode 7, 'A che servono gli amici?', has the RAI English title 'Friends in Need' and the 4Kids English title 'Grounded'.
  Fact 0.55: Episode 7 aired in Italy on 11 February 2004 and in America on 31 July 2004.
  Fact 0.56: The Winx have lost all their magic as punishment for sneaking into Cloudtower.
  Fact 0.57: Griselda tasks the Winx with cleaning the entire school without their magic or technology.
  Fact 0.58: The rest of the students and faculty go to a concert while the Winx clean.
  Fact 0.59: Tecna struggles with no gadgets to help her, causing her to wear a bucket on her head and attempt to use a broom.
  Fact 0.60: Musa, Stella and Bloom have a water fight, wasting their time, energy and supplies.
  Fact 0.61: To make the punishment lighter, the girls secretly invite the boys from Redfountain over after everyone has left.
  Fact 0.62: The girls claim they are throwing an all-night party.
  Fact 0.63: The Trix use the opportunity to sneak into a near-empty Alfea to find the real source of the Dragon Flame.
  Fact 0.64: The boys arrive and learn they will be working as janitors.
  Fact 0.65: The boys still play music as they work.
  Fact 0.66: The Trix teleport to Alfea and summon the vacuums to guide them towards the Dragon Flame.
  Fact 0.67: The Trix distract the gang by summoning a monster.
  Fact 0.68: The Winx and the Specialists meet the distraction.
  Fact 0.69: The boys try to fight with their weapons but are quickly knocked aside.
  Fact 0.70: The Winx, despite having no magic, outsmart the beast by using floor cleaner.
  Fact 0.71: The Trix are unable to find the Dragon Flame and depart to Cloudtower, unseen by anyone.
  Fact 0.72: The Specialists head back to Redfountain.
  Fact 0.73: The next morning, Faragonda sees the damage and hears the story from the Winx.
  Fact 0.74: As a reward for defeating an adversary using only wits and resourcefulness, the Winx's powers are returned to them.
  Fact 0.75: Episode 8, 'La festa della rosa', has the RAI English title 'A Friendship Sundered' and the 4Kids English title 'The Day of the Rose'.
  Fact 0.76: Episode 8 aired in Italy on 13 February 2004 and in America on 7 August 2004.
  Fact 0.77: During the Rose Festival, Bloom and Riven unwittingly fall into a trap by the Trix.
  Fact 0.78: The Trix's trap is meant to divide the close relationship between the Specialists and the Winx.
  Fact 0.79: Riven thinks Darcy saved him in a racing crash.
  Fact 0.80: Riven blames Bloom for what happened in the racing crash.
  Fact 0.81: Riven gets mad at his friends for believing Bloom over him.
  Fact 0.82: Riven transfers out of the squad of Specialists along with Darcy.
  Fact 0.83: Episode 9, 'Il tradimento di Riven', has the RAI English title 'Betrayed!' and the 4Kids English title 'Spelled'.
  Fact 0.84: Episode 9 aired in Italy on 16 February 2004 and in America on 14 August 2004.
  Fact 0.85: There are hostilities between the Winx due to Stella's behavior.
  Fact 0.86: Musa's feelings for Riven are strained.
  Fact 0.87: In a fight with the Trix, the possessor of the Dragon's Flame is revealed to be Bloom.
  Fact 0.88: Episode 10, 'La fiamma del drago', has the RAI English title 'Bloom Tested' and the 4Kids English title 'Magical Reality Check'.
  Fact 0.89: Episode 10 aired in Italy on 18 February 2004 and in America on 21 August 2004.
  Fact 0.90: It is time for Bloom's magical midterm test in a virtual reality machine.
  Fact 0.91: The Trix mess with the computer so they can enter it.
  Fact 0.92: The Trix test Bloom to be sure she is indeed the one who possesses the Dragon Flame.
  Fact 0.93: Episode 11, 'Il regno delle ninfe', has the RAI English title 'The Monster and The Willow' and the 4Kids English title 'Junior League'.
  Fact 0.94: Episode 11 aired in Italy on 20 February 2004 and in America on 28 August 2004.
  Fact 0.95: Flora experiments with her plants to study for her test.
  Fact 0.96: Stella, whose clothes were torn apart by a plant, tells Flora to move the plants.
  Fact 0.97: Tecna, who was attacked by a plant, tells Flora to move the plants.
  Fact 0.98: Flora moves the plants to Black-Mud Swamp.
  Fact 0.99: Flora seeks help from the Water Nymphs in finding a special plant.
  Fact 0.100: The Water Nymphs tell the Winx about a monster that has been terrorizing them.
  Fact 0.101: The Winx decide to fight the monster.
  Fact 0.102: Episode 12, 'Miss Magix', aired in Italy on 23 February 2004 and in America on 4 September 2004.
  Fact 0.103: On the day before a big test, Stella enters a beauty contest.
  Fact 0.104: Lucy from Cloud Tower enters the beauty contest too, with help from the Trix.
  Fact 0.105: The Trix want to play a bad prank on Lucy by first making her pretty.
  Fact 0.106: The Trix plan to sabotage the performances of the other contestants.
  Fact 0.107: The Trix plan to break the spell on Lucy in front of everyone once she has earned the Miss Magix prize.
  Fact 0.108: In the end, Stella wins and keeps her promise to Bloom to study all night for the test the next morning.
  Fact 0.109: Stella makes a few mistakes on the test.
  Fact 0.110: Because Stella studied after the pageant, she passes the test.
  Fact 0.111: Episode 13, 'La figlia del fuoco', has the RAI English title 'A Great Secret Revealed' and the 4Kids English title 'Meant to Be'.
  Fact 0.112: Episode 13 aired in Italy on 25 February 2004 and in America on 11 September 2004.
  Fact 0.113: Bloom returns to Earth during spring break.
  Fact 0.114: Bloom discovers she can read people's auras.
  Fact 0.115: Bloom casts doubts on a pending business deal regarding her mother's flower shop.
  Fact 0.116: A vision was unlocked while Bloom was putting out a fire in her mother's flower shop.
  Fact 0.117: Bloom discovers that Mike found her as a baby while tackling a fire.
  Fact 0.118: Bloom is not Mike and Vanessa's biological daughter.
  Fact 0.119: Episode 14, 'Il segreto di Bloom', has the RAI English title 'Bloom's Dark Secret' and the 4Kids English title 'Witch Trap'.
  Fact 0.120: Episode 14 aired in Italy on 27 February 2004 and in America on 18 September 2004.
  Fact 0.121: Bloom's curiosity about her past and magical powers begins to interfere with her confidence.
  Fact 0.122: Bloom arranges to go to Cloud Tower with Brandon to seek answers.
  Fact 0.123: Bloom meets a kind witch named Mirta.
  Fact 0.124: Episode 15, 'Voci dal passato', has the RAI English title 'Honor Above All' and the 4Kids English title 'Pushing the Envelope'.
  Fact 0.125: Episode 15 aired in Italy on 1 March 2004 and in America on 25 September 2004.
  Fact 0.126: A pending test by Professor Wizgiz worries the Winx Club.
  Fact 0.127: Bloom accidentally finds the answers to the test.
  Fact 0.128: Episode 16, 'Il nemico nell'ombra', has the RAI English title 'Cold Spell' and the 4Kids English title 'The Nightmare Monster'.
  Fact 0.129: Episode 16 aired in Italy on 3 March 2004 and in America on 2 October 2004.
  Fact 0.130: In another attempt to harness the power of the Dragon's Flame, the Trix send a nightmare monster.
  Fact 0.131: The nightmare monster terrorizes the dreams of the Winx.
  Fact 0.132: Episode 17, 'Il segreto di Brandon', has the RAI English title 'Secrets Within Secrets' and the 4Kids English title 'Royal Heartbreak'.
  Fact 0.133: Episode 17 aired in Italy on 5 March 2004 and in America on 9 October 2004.
  Fact 0.134: During festivities at Redfountain, Bloom encounters Princess Diaspro.
  Fact 0.135: Princess Diaspro claims she is betrothed to Brandon, whom she keeps calling Sky.
  Fact 0.136: Bloom finds out that Sky and Brandon switched identities.
  Fact 0.137: Sky and Brandon switched identities because 'Sky' wanted to live a normal life for once.
  Fact 0.138: Episode 18, 'Addio Magix!', has the RAI English title 'The Font of Dragon Fire' and the 4Kids English title 'Senior Witches Go to Earth'.
  Fact 0.139: Episode 18 aired in Italy on 8 March 2004 and in America on 16 October 2004.
  Fact 0.140: Upset about the events at Redfountain, Bloom returns to Earth.
  Fact 0.141: The Trix take advantage of the situation and attack Bloom on Earth.
  Fact 0.142: The witches reveal to Bloom that she is the surviving heir to planet Domino (Sparks).
  Fact 0.143: The witches steal the Dragon Fire from Bloom.
  Fact 0.144: Episode 19, 'Attacco ad Alfea', has the RAI English title 'The Fall of Magix' and the 4Kids English title 'The Army of Decay'.
  Fact 0.145: Episode 19 aired in Italy on 10 March 2004 and in America on 23 October 2004.
  Fact 0.146: Bloom returns to Alfea following the events on Earth.
  Fact 0.147: The Trix take control of Cloud Tower.
  Fact 0.148: The Trix summon the Army of Decay to take control of Magix.
  Fact 0.149: Episode 20, 'La scomparsa di Bloom', has the RAI English title 'Mission to Domino' and the 4Kids English title 'Sparks of Hope'.
  Fact 0.150: Episode 20 aired in Italy on 12 March 2004 and in America on 30 October 2004.
  Fact 0.151: To counter the powerful magic possessed by the Trix, the Winx Club go to planet Domino.
  Fact 0.152: The Winx Club hope to find the Dragon's Flame on planet Domino.
  Fact 0.153: Redfountain is attacked by the Army of Decay.
  Fact 0.154: Episode 21, 'Trappola di ghiaccio', has the RAI English title 'The Crown of Dreams' and the 4Kids English title 'The Frozen Palace'.
  Fact 0.155: Episode 21 aired in Italy on 15 March 2004 and in America on 6 November 2004.
  Fact 0.156: In the ruins of Domino (Sparks), the Winx are attacked during their exploration.
  Fact 0.157: In Magix, the Trix destroy Redfountain.
  Fact 0.158: Episode 22, 'Il ritorno di Riven', has the RAI English title 'Storming Cloudtower' and the 4Kids English title 'Mission to Cloud Tower'.
  Fact 0.159: Episode 22 aired in Italy on 17 March 2004 and in America on 13 November 2004.
  Fact 0.160: The Army of Decay forces the retreat of the population to Alfea.
  Fact 0.161: The Winx go to Cloud Tower to retrieve the power of the Dragon's Flame from the Trix.
  Fact 0.162: Riven rejoins the Specialists.
  Fact 0.163: Episode 23, 'Fuga da Torrenuvola', has the RAI English title 'Power Play' and the 4Kids English title 'The Search for the Flame'.
  Fact 0.164: Episode 23 aired in Italy on 19 March 2004 and in America on 20 November 2004.
  Fact 0.165: The Trix torment the Winx for being unable to find the power of the Dragon's Flame on Domino (Sparks).
  Fact 0.166: The cadre and staff of Cloud Tower escape to Alfea.
  Fact 0.167: Episode 24, 'Il mistero del lago', has the RAI English title 'The Witches' Siege' and the 4Kids English title 'Battle for Alfea'.
  Fact 0.168: Episode 24 aired in Italy on 22 March 2004 and in America on 27 November 2004.
  Fact 0.169: The Army of Darkness launches an attack on Alfea.
  Fact 0.170: Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice.
  Fact 0.171: Episode 25, 'Il sonno di Magix', has the RAI English title 'The Ultimate Challenge' and the 4Kids English title 'The Great Witch Invasion'.
  Fact 0.172: Episode 25 aired in Italy on 24 March 2004 and in America on 4 December 2004.
  Fact 0.173: The first attack on Alfea fails.
  Fact 0.174: Bloom follows the spirit of her older sister Daphne.
  Fact 0.175: Bloom regains her Dragon Fire powers.
  Fact 0.176: The Trix lead the Army of Decay in a final attack on Alfea.
  Fact 0.177: Episode 26, 'Battaglia finale', has the RAI English title 'The Witches' Downfall' and the 4Kids English title 'Fire and Ice'.
  Fact 0.178: Episode 26 aired in Italy on 26 March 2004 and in America on 11 December 2004.
  Fact 0.179: The Army of Darkness approaches Alfea.
  Fact 0.180: The Trix seek to take complete control of the Magix Dimension.
  Fact 0.181: The Winx, except Bloom, take action to fend off the Trix.
  Fact 0.182: Alfea's students and teachers take action to fend off the Trix.
  Fact 0.183: They fend off the Trix to keep the Trix from completely destroying algae to the ground before the Trix use the powers of the Dragon.
  Fact 0.184: RC1 is the Italian title and the RAI English title of the next episode.
  Fact 0.185: The 4Kids English title of RC1 is 'Attack of the Senior Witches'.
  Fact 0.186: RC1 has no Italian air date.
  Fact 0.187: RC1 aired in America on 22 May 2004.
  Fact 0.188: RC1 is a clip show episode made by 4Kids Entertainment.
  Fact 0.189: RC1 features scenes from episodes 18, 19 and 20.
  Fact 0.190: The scenes in RC1 show the Trix attacking Alfea.
  Fact 0.191: RC1 serves as a sneak preview to the first season.

Retrieved items:
<item 1>
Winx Club (season 1)
18 18 `` Addio Magix ! '' / `` The Font of Dragon Fire '' `` Senior Witches Go to Earth '' 8 March 2004 ( 2004 - 03 - 08 ) 16 October 2004 Upset about the
events at Redfountain , Bloom returns to Earth . The Trix take advantage of the situation and attack her on Earth . The witches reveal to Bloom that she is the
surviving heir to planet Domino ( Sparks ) , and they steal the Dragon Fire from her . 19 19 `` Attacco ad Alfea '' / `` The Fall of Magix '' `` The Army of
Decay '' 10 March 2004 ( 2004 - 03 - 10 ) 23 October 2004 Bloom returns to Alfea following the events on Earth . The Trix take control of Cloud Tower and summon
the Army of Decay to take control of Magix .
</item 1>

<item 2>
Winx Club (season 1)
Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03
) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17
`` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at
Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon
switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! ''
</item 2>

<item 3>
Winx Club (season 1)
10 10 `` La fiamma del drago '' / `` Bloom Tested '' `` Magical Reality Check '' 18 February 2004 ( 2004 - 02 - 18 ) 21 August 2004 It 's time for Bloom 's
magical midterm test in a virtual reality machine . The Trix mess with the computer so they can enter it and test her to be sure of Bloom is indeed the one who
possesses the Dragon Flame . 11 11 `` Il regno delle ninfe '' / `` The Monster and The Willow '' `` Junior League '' 20 February 2004 ( 2004 - 02 - 20 ) 28
August 2004 Flora experiments with her plants to study for her test . Stella ( who had her clothes torn apart by a plant ) and Tecna ( who was attacked by a
plant ) tell Flora to move the plants . She moves them to Black - Mud Swamp . Flora seeks help from the Water Nymphs in finding a special plant . The Water
Nymphs tell the Winx about a monster which has been terrorizing them and the Winx decide to fight it .
</item 3>

<item 4>
Winx Club (season 1)
Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice . 25 25 `` Il sonno di Magix '' / `` The Ultimate Challenge '' `` The Great Witch
Invasion '' 24 March 2004 ( 2004 - 03 - 24 ) 4 December 2004 The first attack on Alfea fails . Bloom follows the spirit of her older - sister Daphne and regains
her Dragon Fire powers . The Trix lead the Army of Decay in a final attack on Alfea . 26 26 `` Battaglia finale '' / `` The Witches ' Downfall '' `` Fire and
Ice '' 26 March 2004 ( 2004 - 03 - 26 ) 11 December 2004 As the Army of Darkness approaching Alfea and the Trix seeking to take complete control of the Magix
Dimension , the Winx , except Bloom and including Alfea 's students and teachers take action to fend off the Trix from completely destroying algae to the ground
before the Trix use the powers of the Dragon .
</item 4>

<item 5>
Winx Club (season 1)
Later Bloom spots the Trix trying to steal Stella 's ring , and tries to fight them off , transforming in a fairy for the first time . Although the Trix are
able to escape , the other Winx girls inform Bloom that Stella 's ring is safe and that they swapped the contents of the case with a baby duck ( Pepe ) that
bonds with Icy . `` La palude di Melmamora '' / `` The Black - Mud Swamp '' `` The Voice of Nature '' 4 February 2004 ( 2004 - 02 - 04 ) 10 July 2004 As part of
Professor Palladium 's field trip , the class must navigate the Black - Mud Swamp to reach a clearing in the Gloomy - Wood Forest by listening to nature and
without using any magic . The boys from Red Fountain are transporting a troll prisoner but the Trix cause their aircraft to crash land , and the troll is freed
. The Winx and the Red Fountain boys end up meeting in the swamp , and must work together , despite their differences , to find their way out .
</item 5>

<item 6>
Winx Club (season 1)
A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning
Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 -
02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower
with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March
2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test .
</item 6>

<item 7>
Winx Club (season 1)
12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud
Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other
contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom
, which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13
13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth
during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop .
</item 7>

<item 8>
Winx Club (season 1)
20 20 `` La scomparsa di Bloom '' / `` Mission to Domino '' `` Sparks of Hope '' 12 March 2004 ( 2004 - 03 - 12 ) 30 October 2004 To counter the powerful magic
possessed by the Trix , the Winx Club go to planet Domino hoping to find the Dragon 's Flame . Redfountain is attacked by the Army of Decay . 21 21 `` Trappola
di ghiaccio '' / `` The Crown of Dreams '' `` The Frozen Palace '' 15 March 2004 ( 2004 - 03 - 15 ) 6 November 2004 In the ruins of Domino ( Sparks ) , the Winx
are attacked during their exploration . In Magix , Redfountain is destroyed by the Trix . 22 22 `` Il ritorno di Riven '' / `` Storming Cloudtower '' `` Mission
to Cloud Tower '' 17 March 2004 ( 2004 - 03 - 17 ) 13 November 2004 The Army of Decay forces the retreat of the population to Alfea .
</item 8>

<item 9>
Winx Club (season 1)
The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria
! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic
energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms
out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower
'' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out
what made the Trix go to such lengths to get it .
</item 9>

<item 10>
Winx Club (season 1)
Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head
Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello
di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school ,
Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded
. She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix
casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball .
</item 10>
=== END ITEM id=c2aded53e1a0a0b0 ===

=== ITEM id=0e6d11556dd6212d ===
Gold sentences and facts:
Sentence 0: Transcription occurs in the cell nucleus , where the DNA is held . The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held together by hydrogen bonds between the bases of opposite strands . The sugar and the phosphate in each strand are joined together by stronger phosphodiester covalent bonds . The DNA is `` unzipped '' ( disruption of hydrogen bonds between different single strands ) by the enzyme helicase , leaving the single nucleotide chain open to be copied . RNA polymerase reads the DNA strand from the 3 - prime ( 3 ' ) end to the 5 - prime ( 5 ' ) end , while it synthesizes a single strand of messenger RNA in the 5 ' - to - 3 ' direction . The general RNA structure is very similar to the DNA structure , but in RNA the nucleotide uracil takes the place that thymine occupies in DNA . The single strand of mRNA leaves the nucleus through nuclear pores , and migrates into the cytoplasm .
  Fact 0.0: Transcription occurs in the cell nucleus.
  Fact 0.1: The DNA is held in the cell nucleus.
  Fact 0.2: The DNA structure of the cell is made up of two helixes.
  Fact 0.3: Each helix of the DNA structure is made up of sugar and phosphate.
  Fact 0.4: The two helixes of the DNA structure are held together by hydrogen bonds between the bases of opposite strands.
  Fact 0.5: The sugar and the phosphate in each strand of DNA are joined together by phosphodiester covalent bonds.
  Fact 0.6: The phosphodiester covalent bonds are stronger than the hydrogen bonds.
  Fact 0.7: The DNA is "unzipped" by the enzyme helicase.
  Fact 0.8: The unzipping of the DNA is disruption of hydrogen bonds between different single strands.
  Fact 0.9: Unzipping the DNA leaves the single nucleotide chain open to be copied.
  Fact 0.10: RNA polymerase reads the DNA strand from the 3' end to the 5' end.
  Fact 0.11: While reading the DNA strand, RNA polymerase synthesizes a single strand of messenger RNA.
  Fact 0.12: RNA polymerase synthesizes the messenger RNA in the 5' to 3' direction.
  Fact 0.13: The general RNA structure is very similar to the DNA structure.
  Fact 0.14: In RNA, the nucleotide uracil takes the place that thymine occupies in DNA.
  Fact 0.15: The single strand of messenger RNA leaves the nucleus through nuclear pores.
  Fact 0.16: The single strand of messenger RNA migrates into the cytoplasm.

Retrieved items:
<item 1>
Protein biosynthesis
The first product of transcription differs in prokaryotic cells from that of eukaryotic cells , as in prokaryotic cells the product is mRNA , which needs no
post-transcriptional modification , whereas , in eukaryotic cells , the first product is called primary transcript , that needs post-transcriptional
modification ( capping with 7 - methyl - guanosine , tailing with a poly A tail ) to give hnRNA ( heterogeneous nuclear RNA ) . hnRNA then undergoes splicing of
introns ( noncoding parts of the gene ) via spliceosomes to produce the final mRNA . Translation ( edit ) Main article : Translation ( biology ) Diagram showing
the process of translation Diagram showing the translation of mRNA and the synthesis of proteins by a ribosome Phenomena of amino acid assembly from RNA . The
synthesis of proteins from RNA is known as translation . In eukaryotes , translation occurs in the cytoplasm , where the ribosomes are located . Ribosomes are
made of a small and large subunit that surround the mRNA .
</item 1>

<item 2>
Protein biosynthesis
Ribosomes are made of a small and large subunit that surround the mRNA . In translation , messenger RNA ( mRNA ) is decoded to produce a specific polypeptide
according to the rules specified by the trinucleotide genetic code . This uses an mRNA sequence as a template to guide the synthesis of a chain of amino acids
that form a protein . Translation proceeds in four phases : activation , initiation , elongation , and termination ( all describing the growth of the amino acid
chain , or polypeptide that is the product of translation ) . In activation , the correct amino acid ( AA ) is joined to the correct transfer RNA ( tRNA ) .
While this is not , in the technical sense , a step in translation , it is required for translation to proceed . The AA is joined by its carboxyl group to the 3
' OH of the tRNA by an ester bond . When the tRNA has an amino acid linked to it , it is termed `` charged '' . Initiation involves the small subunit of the
ribosome binding to 5 ' end of mRNA with the help of initiation factors ( IF ) , other proteins that assist the process .
</item 2>

<item 3>
Protein biosynthesis
at 22 : 32 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Protein biosynthesis
Protein biosynthesis - wikipedia Protein biosynthesis Jump to : navigation , search RNA is transcribed in the nucleus ; once completely processed , it is
transported to the cytoplasm and translated by the ribosome ( shown in very pale grey behind the tRNA ) . Protein synthesis is the process whereby biological
cells generate new proteins ; it is balanced by the loss of cellular proteins via degradation or export . Translation , the assembly of amino acids by ribosomes
, is an essential part of the biosynthetic pathway , along with generation of messenger RNA ( mRNA ) , aminoacylation of transfer RNA ( tRNA ) ,
co-translational transport , and post-translational modification . Protein biosynthesis is strictly regulated at multiple steps . They are principally during
transcription ( phenomena of RNA synthesis from DNA template ) and translation ( phenomena of amino acid assembly from RNA ) . The cistron DNA is transcribed
into the first of a series of RNA intermediates . The last version is used as a template in synthesis of a polypeptide chain . Protein will often be synthesized
directly from genes by translating mRNA .
</item 4>

<item 5>
Protein biosynthesis
Transcription occurs in the cell nucleus , where the DNA is held . The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held
together by hydrogen bonds between the bases of opposite strands . The sugar and the phosphate in each strand are joined together by stronger phosphodiester
covalent bonds . The DNA is `` unzipped '' ( disruption of hydrogen bonds between different single strands ) by the enzyme helicase , leaving the single
nucleotide chain open to be copied . RNA polymerase reads the DNA strand from the 3 - prime ( 3 ' ) end to the 5 - prime ( 5 ' ) end , while it synthesizes a
single strand of messenger RNA in the 5 ' - to - 3 ' direction . The general RNA structure is very similar to the DNA structure , but in RNA the nucleotide
uracil takes the place that thymine occupies in DNA . The single strand of mRNA leaves the nucleus through nuclear pores , and migrates into the cytoplasm .
</item 5>

<item 6>
Protein biosynthesis
This whole complex of processes is carried out by the ribosome , formed of two main chains of RNA , called ribosomal RNA ( rRNA ) , and more than 50 different
proteins . The ribosome latches onto the end of an mRNA molecule and moves along it , capturing loaded tRNA molecules and joining together their amino acids to
form a new protein chain . Protein biosynthesis , although very similar , is different for prokaryotes and eukaryotes . Contents ( hide ) 1 Transcription 2
Translation 3 Events during or following protein translation 4 See also 5 References 6 External links Transcription ( edit ) Main article : Transcription (
genetics ) Diagram showing the process of transcription Phenomena of RNA synthesis from DNA template In transcription an mRNA chain is generated , with one
strand of the DNA double helix in the genome as a template . This strand is called the template strand . Transcription can be divided into 3 stages : initiation
, elongation , and termination , each regulated by a large number of proteins such as transcription factors and coactivators that ensure that the correct gene
is transcribed . Transcription occurs in the cell nucleus , where the DNA is held .
</item 6>

<item 7>
Protein biosynthesis
Elongation occurs when the next aminoacyl - tRNA ( charged tRNA ) in line binds to the ribosome along with GTP and an elongation factor . Termination of the
polypeptide happens when the A site of the ribosome faces a stop codon ( UAA , UAG , or UGA ) . When this happens , no tRNA can recognize it , but releasing
factor can recognize nonsense codons and causes the release of the polypeptide chain . The capacity of disabling or inhibiting translation in protein
biosynthesis is used by some antibiotics such as anisomycin , cycloheximide , chloramphenicol , tetracycline , streptomycin , erythromycin , puromycin , etc .
Events during or following protein translation ( edit ) Main articles : Proteolysis , Posttranslational modification , and Protein folding Events that occur
during or following biosynthesis include proteolysis , post-translational modification and protein folding . Proteolysis may remove N - terminal , C - terminal
or internal amino - acid residues or peptides from the polypeptide .
</item 7>

<item 8>
Protein biosynthesis
Jump up ^ Alberts , Bruce . Molecular Biology of the Cell , 5e . New York : Garland Science , 2008 . External links ( edit ) Interactive Java simulation of
transcription initiation . From Center for Models of Life at the Niels Bohr Institute . A website focused primarily on Protein Synthesis . Protein primary
structure and posttranslational modifications General Peptide bond Protein biosynthesis Proteolysis Racemization N-O acyl shift N terminus Acetylation
Carbamylation Formylation Glycation Methylation Myristoylation ( Gly ) C terminus Amidation Glycosyl phosphatidylinositol ( GPI ) O - methylation Detyrosination
Single specific AAs Serine / Threonine Phosphorylation Dephosphorylation Glycosylation Methylidene - imidazolone ( MIO ) formation Tyrosine Phosphorylation
Dephosphorylation Sulfation Porphyrin ring linkage Adenylylation Flavin linkage Topaquinone ( TPQ ) formation Detyrosination Cysteine Palmitoylation Prenylation
Aspartate Succinimide formation Glutamate Carboxylation Methylation
</item 8>

<item 9>
Protein biosynthesis
Protein will often be synthesized directly from genes by translating mRNA . However , when a protein must be available on short notice or in large quantities ,
a protein precursor is produced . A proprotein is an inactive protein containing one or more inhibitory peptides that can be activated when the inhibitory
sequence is removed by proteolysis during posttranslational modification . A preprotein is a form that contains a signal sequence ( an N - terminal signal
peptide ) that specifies its insertion into or through membranes , i.e. , targets them for secretion . The signal peptide is cleaved off in the endoplasmic
reticulum . Preproproteins have both sequences ( inhibitory and signal ) still present . In protein synthesis , a succession of tRNA molecules charged with
appropriate amino acids are brought together with an mRNA molecule and matched up by base - pairing through the anti-codons of the tRNA with successive codons
of the mRNA . The amino acids are then linked together to extend the growing protein chain , and the tRNAs , no longer carrying amino acids , are released .
</item 9>

<item 10>
Protein biosynthesis
The termini and side - chains of the polypeptide may be subjected to post-translational modification . These modifications may be required for correct cellular
localisation or the natural function of the protein . During and after synthesis , polypeptide chains often fold to assume , so called , native secondary and
tertiary structures . This is known as protein folding and is typically required for the natural function of the protein . See also ( edit ) Central dogma of
molecular biology Cistron Gene expression Genetic code Operon Peptide synthesis Protein production References ( edit ) Jump up ^ Kafri M , Metzl - Raz E , Jona
G , Barkai N. 2016 . The Cost of Protein Production . Cell Rep 14 : 22 -- 31 . https://dx.doi.org/10.1016/j.celrep.2015.12.015 ^ Jump up to : Alberts , Bruce (
2002 ) . Molecular biology of the cell . New York : Garland Science . p. 760 . ISBN 0 - 8153 - 3218 - 1 . Jump up ^ Alberts , Bruce .
</item 10>
=== END ITEM id=0e6d11556dd6212d ===

=== ITEM id=7468a0dfe709a694 ===
Gold sentences and facts:
Sentence 0: Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 - 13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
  Fact 0.0: The table lists 17 episodes with first airing on September 13, 1969 and last airing on January 17, 1970.
  Fact 0.1: The table lists 8 episodes with first airing on September 12, 1970 and last airing on October 31, 1970.
  Fact 0.2: The table lists 16 episodes with first airing on September 9, 1978 and last airing on December 23, 1978.

Retrieved items:
<item 1>
List of Scooby-Doo, Where Are You! episodes
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
List of Scooby-Doo, Where Are You! episodes
24 7 `` Who 's Afraid of the Big Bad Werewolf ? '' Ghost of Silas Long the Werewolf / Sheep Rustler Tell Me , Tell Me October 24 , 1970 ( 1970 - 10 - 24 ) Out
on a camping trip , the gang hears the sounds of a wolf , so they decide to follow wolf - like tracks to the open and empty grave of Silas Long , who was
supposedly a werewolf . They follow his tracks further to an old abandoned sawmill filled with wool , empty barrels , and the werewolf ghost himself . 25 8 ``
Do n't Fool with a Phantom '' Wax Phantom / Roger Steven Pretty Mary Sunlight October 31 , 1970 ( 1970 - 10 - 31 ) While participating in the Johnny Sands Dance
Game Show at the local TV station , the teens encounter a phantom made of wax , who steals a safe full of money and kidnaps the station manager . Is the phantom
a hoax , or the result of a curse from an old station employee seeking revenge for the cancellation of his show ?
</item 2>

<item 3>
List of Scooby-Doo, Where Are You! episodes
The Willawaw ! '' The Williwaw and Owl Men / Grey Fox and henchmen September 9 , 1978 ( 1978 - 09 - 09 ) While visiting Velma 's Uncle Dave , the gang find out
he has been kidnapped from his home , and that a mysterious Williwaw is haunting the area . 27 `` A Creepy Tangle in the Bermuda Triangle '' Skeleton Men / Dr.
Grimsley September 16 , 1978 ( 1978 - 09 - 16 ) While on a boat flowing down the Gulf Stream , the gang end up in the Bermuda Triangle and discover an airplane
- snatching flying saucer and a trio of Skeleton Men . 28 `` A Scary Night with a Snow Beast Fright '' Snow Beast / Dr. Bapiste September 23 , 1978 ( 1978 - 09
- 23 ) After being invited by Professor Krueger at the North Pole , the gang finds out that he has been captured . After they meet an Eskimo named Chief Minook
, he gets captured as well , and the gang attempts to solve the mystery .
</item 3>

<item 4>
List of Scooby-Doo, Where Are You! episodes
Wickles September 13 , 1969 ( 1969 - 09 - 13 ) While walking home from the movies , Shaggy and Scooby - Doo discover a black suit of armor in a pickup truck ,
and the gang deliver it to the local museum . The kids later break into the museum to search for clues , after learning that the archaeologist who was
transporting the suit is missing , and they find that the knight 's armor has come to life . `` Hassle in the Castle '' Phantom of Vasquez / Bluestone the Great
September 20 , 1969 ( 1969 - 09 - 20 ) The gang goes out on a boating expedition , but gets lost in a fog and crashes their boat on the mysterious Haunted Isle
. They decide to venture to the abandoned Vázquez Castle on the island for help , but instead encounter a transparent phantom running loose , who threatens them
to leave the island and never return .
</item 4>

<item 5>
List of Scooby-Doo, Where Are You! episodes
5 5 `` Decoy for a Dognapper '' Indian Witch Doctor and Ghost of Geronimo / Buck Masters October 11 , 1969 ( 1969 - 10 - 11 ) When a series of local dognappings
alarms prize - winning dog trainer Buck Masters , the gang offers to have Scooby - Doo pose as a decoy to catch the dognappers . However , their plan works too
well as Scooby is himself dognapped , leading the gang to discover that the dognappers appear to be under the command of a ghostly Indian witch doctor . 6 6 ``
What the Hex Going On ? '' Ghost of Elias Kingston / Stuart Weatherby October 18 , 1969 ( 1969 - 10 - 18 ) The kids pay a visit to their friend Sharon Weatherby
, whose Uncle Stuart has been cursed by the ghost of the family 's neighbor , Elias Kingston . When Uncle Stuart disappears altogether , Mystery , Inc. ventures
into the abandoned and allegedly haunted Kingston Mansion next door , in search of him and the ghost .
</item 5>

<item 6>
List of Scooby-Doo, Where Are You! episodes
List of Scooby - Doo , where are You ! Episodes - wikipedia List of Scooby - Doo , where are You ! Episodes Jump to : navigation , search This article does not
cite any sources . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2012
) ( Learn how and when to remove this template message ) Scooby - Doo , Where Are You ! is a Saturday morning cartoon produced by Hanna - Barbera .
</item 6>

<item 7>
List of Scooby-Doo, Where Are You! episodes
is a Saturday morning cartoon produced by Hanna - Barbera . Contents ( hide ) 1 Series overview 2 Episodes 2.1 Season 1 ( 1969 -- 70 ) 2.2 Season 2 ( 1970 ) 2.3
Season 3 ( 1978 ) ( Revival ) 3 References Series overview ( edit ) Season Episodes Originally aired First aired Last aired 17 September 13 , 1969 ( 1969 - 09 -
13 ) January 17 , 1970 ( 1970 - 01 - 17 ) 8 September 12 , 1970 ( 1970 - 09 - 12 ) October 31 , 1970 ( 1970 - 10 - 31 ) 16 September 9 , 1978 December 23 , 1978
Episodes ( edit ) Season 1 ( 1969 -- 70 ) ( edit ) No . overall No. in season Title Villain ( s ) Original air date `` What a Night for a Knight '' Black Knight
/ Mr.
</item 7>

<item 8>
List of Scooby-Doo, Where Are You! episodes
banner , as the revival series was cancelled . 35 10 `` The Creepy Creature of Vulture 's Claw '' Mantis Creature / Professor Greer November 11 , 1978 ( 1978 -
11 - 11 ) The gang discovers a humanoid mantis creature roaming around Vulture 's Claw Botanical Gardens . 36 11 `` The Diabolical Disc Demon '' The Phantom /
Ace Decade November 18 , 1978 ( 1978 - 11 - 18 ) The kids discovers the Diabolical Disc Demon . He wants a half - finished song of a missing writer and will
stop at nothing to get it . 37 12 `` Scooby 's Chinese Fortune Kooky Caper '' Moon Monster / Uncle Chin November 25 , 1978 ( 1978 - 11 - 25 ) The gang runs into
The Moon Monster . There are rumors that if you run into his shadow , you will be turned into stone .
</item 8>

<item 9>
List of Scooby-Doo, Where Are You! episodes
21 `` Scooby 's Night with a Frozen Fright '' Caveman / Professor Wayne Seven Days a Week October 3 , 1970 ( 1970 - 10 - 03 ) While out fishing , Shaggy and
Scooby - Doo hook a two - million - year - old caveman frozen in a block of ice , which the gang learns was lost at sea during a violent storm . They take the
caveman to the Oceanland aquarium , but when someone melts the ice block and the caveman goes on the rampage , the gang decides to stick around and investigate
. 22 5 `` Haunted House Hang - Up '' Headless Spectre / Penrod Stillwall , Phantom / Asa Shanks Love the World October 10 , 1970 ( 1970 - 10 - 10 ) The Mystery
Machine overheats in front of an old spooky mansion , and the gang goes to look for help , but finds out that the house is being haunted by the Headless Spectre
, a terrifying phantom with no head . The mystery gets even stranger when they begin to hear knocking sounds and clues leading to a treasure .
</item 9>

<item 10>
List of Scooby-Doo, Where Are You! episodes
This is initially the series finale and the last episode under the Scooby - Doo , Where Are You ! name until September , 1978 . Season 3 ( 1978 ) ( revival ) (
edit ) Note : Only the first nine episodes of the sixteen episode 1978 season of Scooby - Doo , listed below , were broadcast under the Scooby - Doo , Where are
You ! name . The revival series was canceled afterward , and the remaining seven episodes , that were intended for the revival , were instead broadcast during
the Scooby 's All - Stars block , and all sixteen were rerun during this block and later syndicated as part of The Scooby - Doo Show . See List of The Scooby -
Doo Show episodes for the entire season run and The Scooby - Doo Show for proper production credits . No . overall No. in season Title Villain ( s ) Original
air date 26 `` Watch Out ! The Willawaw ! ''
</item 10>
=== END ITEM id=7468a0dfe709a694 ===
