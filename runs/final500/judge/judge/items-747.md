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

=== ITEM id=7c1cacd8b97221ea ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 1>

<item 2>
Anne (TV series)
The series premiered on March 19 , 2017 , on CBC , the season finale airing on April 30 , 2017 . On August 3 , 2017 , both CBC and Netflix renewed the series
for a 10 - episode second season to premiere in 2018 . Season 2 began production in November 2017 . Contents ( hide ) 1 Premise 2 Cast 2.1 Main 2.2 Others 3
Production 3.1 Personnel 3.2 Casting 3.3 Filming locations 3.4 Music 4 Themes 5 Broadcast and release 6 Episodes 6.1 Season 1 ( 2017 ) 7 Reception 7.1 Critical
response 8 References 9 External links Premise ( edit ) Brother and sister Matthew and Marilla Cuthbert , both in late middle age ( Matthew is 60 ) , decide to
take on an orphan boy to help out around their ancestral farm of Green Gables , on the outskirts of the town of Avonlea , on Prince Edward Island .
</item 2>

<item 3>
Anne (TV series)
of seasons No. of episodes 7 ( list of episodes ) Production Executive producer ( s ) Elizabeth Bradley Alex Sapot Sally Catto Debra Hayward Alison Owen Miranda
de Pencier Moira Walley - Beckett Producer ( s ) Susan Murdoch Cinematography Bobby Shore Running time 44 minutes Production company ( s ) Pelican Ballet
Northwood Entertainment Distributor Netflix Release Original network CBC Television Netflix ( worldwide ) Picture format 4K ( Ultra HD ) Audio format Dolby
Digital 5.1 with Descriptive Video Service track Original release March 19 , 2017 ( 2017 - 03 - 19 ) -- present ( present ) External links Website Anne ( titled
Anne with an E for international distribution on Netflix ) is a Canadian drama television series based on the 1908 novel Anne of Green Gables by Lucy Maud
Montgomery , and adapted by Emmy Award - winning writer and producer Moira Walley - Beckett . It airs on CBC in Canada , and elsewhere in the world it is
available for streaming on Netflix . The first season consists of seven episodes , with Niki Caro directing the 90 - minute season premiere .
</item 3>

<item 4>
Anne (TV series)
Numeris . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 10 , 2017 ) . `` SUN onites SNET NHL TOR - COL 1181k VAN -- EDM 766k TSN CURL 1026k GLO NCISLA 1Mk
CBC ANNE 654k STORY US 546k CITY SHOTS 623k CTV SAVE HOPE 548k '' ( Tweet ) -- via Twitter . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 24 , 2017 ) . ``
SJN o'nites SNet NHL TOR - WASH Game 6 3467k OTT - BOS 1284k OMNI2 TOR - WASH 121k CTV SAVE HOPE 686K CBC ANNE 656K STORY OF US 450K '' ( Tweet ) -- via Twitter
. Jump up ^ `` Anne with an E ( 2017 - ) '' . Rotten Tomatoes . Retrieved 4 January 2018 . Jump up ^ `` Anne with an E : Season 1 '' . Metacritic .
</item 4>

<item 5>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 5>

<item 6>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 6>

<item 7>
Anne (TV series)
IndieWire . Retrieved 17 June 2017 . Jump up ^ Chaney , Jen . `` Anne of Green Gables Fans , You Will Love Netflix 's Anne With an E '' . Vulture.com . New York
Magazine . Retrieved 17 June 2017 . Jump up ^ Genzlinger , Neil ( May 11 , 2017 ) . `` Review : ' Anne With an E ' Is a Rewarding Return to Green Gables '' .
The New York Times . Retrieved 17 June 2017 . Jump up ^ Hirschlag , Annie . `` ' Anne with an E ' Review : The Neflix Adpatation Hits the Nostalgia Button and
More '' . Mic.com . Mic ( media Company ) . Retrieved 17 June 2017 . Jump up ^ Nawaz , Saleema ( March 21 , 2017 ) . `` We asked an Anne of Green Gables
superfan to review the first episode of CBC 's new TV series '' . Toronto Life . Retrieved 4 June 2017 .
</item 7>

<item 8>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 8>

<item 9>
Anne (TV series)
Jump up ^ `` Anne with an E : Season 1 '' . Metacritic . Retrieved 17 June 2017 . Jump up ^ Ashby , Emily . `` Anne with an E '' . Common Sense Media .
Retrieved 17 June 2017 . Jump up ^ Cerny , Tasha . `` ANNE WITH AN E Review : `` Episodes 1 - 3 '' `` . www.thetrackingboard.com . Retrieved 18 June 2017 . Jump
up ^ Inhat , Gwen . `` Anne With An E offers a winning , darker take on a familiar tale '' . A.V. Club . Retrieved 18 June 2017 . Jump up ^ Schneller , Johanna
( March 20 , 2017 ) . `` This Anne of Green Gables hints at miseries : What I 'm Watching '' . Toronto Star . access - date = requires url = ( help ) Jump up ^
Nguyen , Hanh . `` ' Anne with an E ' Review : A ' Breaking Bad ' Producer Updates ' Anne of Green Gables ' for Our Darker Times '' .
</item 9>

<item 10>
Anne (TV series)
`` CBC and Netflix renew reimagined Anne of Green Gables '' . Toronto Star . Retrieved August 3 , 2017 . Jump up ^ `` Anne ( with an E ) : Season Two Filming
Begins for CBC and Netflix Series '' . TV Series Finale . November 24 , 2017 . ^ Jump up to : `` Netflix 's ' Anne of Green Gables ' Adaptation Finds Its Anne
Shirley , Casts 2 Other Roles '' . Deadline . October 21 , 2016 . ^ Jump up to : `` Netflix 's ' Anne Of Green Gables ' Adaptation Adds Three To Cast '' .
Deadline.com . Retrieved November 21 , 2016 . Jump up ^ `` Maureen McCormick To Guest In ' The Guest Book ' ; Lucas Jade Zumann Joins ' Anne ' '' . Deadline.com
. Retrieved November 18 , 2016 . Jump up ^ Hunt , Nigel ( March 19 , 2017 ) . `` ' Ai n't your grandmother 's Anne ' : new series gives gritty Green Gables amid
glut of Anne adaptations '' . CBC News .
</item 10>
=== END ITEM id=7c1cacd8b97221ea ===

=== ITEM id=10acf7638d5b1943 ===
Gold sentences and facts:
Sentence 0: `` Go Tell It on the Mountain '' is an African - American spiritual song , compiled by John Wesley Work , Jr. , dating back to at least 1865 , that has been sung and recorded by many gospel and secular performers . It is considered a Christmas carol because its original lyrics celebrate the Nativity of Jesus :
  Fact 0.0: 'Go Tell It on the Mountain' is an African-American spiritual song.
  Fact 0.1: 'Go Tell It on the Mountain' was compiled by John Wesley Work, Jr.
  Fact 0.2: 'Go Tell It on the Mountain' dates back to at least 1865.
  Fact 0.3: 'Go Tell It on the Mountain' has been sung and recorded by many gospel performers.
  Fact 0.4: 'Go Tell It on the Mountain' has been sung and recorded by many secular performers.
  Fact 0.5: 'Go Tell It on the Mountain' is considered a Christmas carol.
  Fact 0.6: The original lyrics of 'Go Tell It on the Mountain' celebrate the Nativity of Jesus.
  Fact 0.7: 'Go Tell It on the Mountain' is considered a Christmas carol because its original lyrics celebrate the Nativity of Jesus.

Retrieved items:
<item 1>
Go Tell It on the Mountain (song)
Go Tell It on the Mountain ( song ) - wikipedia Go Tell It on the Mountain ( song ) Jump to : navigation , search Go Tell It on the Mountain by unknown Genre
Christmas , spiritual Language English Published 1909 `` Go Tell It on the Mountain '' is an African - American spiritual song , compiled by John Wesley Work ,
Jr. , dating back to at least 1865 , that has been sung and recorded by many gospel and secular performers . It is considered a Christmas carol because its
original lyrics celebrate the Nativity of Jesus : `` Go tell it on the mountain , over the hills and everywhere ; go tell it on the mountain , that Jesus Christ
is born . '' Recording artists ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed .
</item 1>

<item 2>
Go Tell It on the Mountain (song)
Christmas Christmas Eve Children 's Day Boxing Day Nochebuena Saint Nicholas Day St. Stephen 's Day Sol Invictus Yule In Christianity Biblical Magi Adoration of
the Magi Adoration of the Shepherds Advent Angel Gabriel Annunciation Annunciation to the shepherds Baptism of the Lord Bethlehem Christingle Christmastide
Epiphany Herod the Great Jesus Joseph Mary Massacre of the Innocents flight into Egypt Nativity Fast Nativity of Jesus in art in later culture Nativity scene
Saint Nicholas Star of Bethlehem Twelfth Night In folklore Badalisc La Befana Belsnickel Caganer Christkind Ded Moroz Elves Father Christmas Grýla Jack Frost
Joulupukki Knecht Ruprecht Korvatunturi Krampus Mikulás Miner 's figure Mrs. Claus Nisse / Tomte North Pole Old Man Winter Olentzero Père Fouettard Père Noël
Rudolph the Red - Nosed Reindeer Saint Lucy Santa 's reindeer Santa 's workshop Sinterklaas Tió de Nadal Vertep Yule Cat Yule Lads Zwarte Piet Gift - bringers
Saint Nicholas Santa Claus List of
</item 2>

<item 3>
Go Tell It on the Mountain (song)
Dumpling Hallaca Tamale Meat Ham Roast goose Romeritos Turkey Stuffing Portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Go_Tell_It_on_the_Mountain_(song)&oldid=812507828 '' Categories : Christmas carols Songs about mountains Songs about
Jesus Mahalia Jackson songs Peter , Paul and Mary songs Hidden categories : Articles needing additional references from December 2016 All articles needing
additional references Talk About Wikipedia Deutsch Italiano Svenska Tagalog Edit links This page was last edited on 28 November 2017 , at 06 : 07 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Go Tell It on the Mountain (song)
on the album ( various artists ) 2003 : The Blind Boys of Alabama on the album Go Tell It on the Mountain 2004 : James Taylor on the album A Christmas Album
2007 : Toby Keith on the album A Classic Christmas 2008 : Needtobreathe on the album An Alternative Christmas ( various artists ) 2008 : Sheryl Crow on the
album Home for Christmas 2009 : Big Daddy Weave on the album Christ Is Come 2012 : Pentatonix on the album PTXmas 2013 : David Crowder Band on the album Wow
Christmas : 30 Top Christian Artists and Holiday Songs 2015 : Hunter Hayes on the album Country Faith Christmas ( various artists ) 2016 : Jennifer Nettles on
the album To Celebrate Christmas 2016 : Sarah McLachlan on the album Wonderland 2016 : Blues Traveler on Go Tell It On The Mountain ( single ) 2017 : Hanson (
band ) - mashup of ' Joy To The World ' and ' Go Tell It On The Mountain ' ( ' Joy To The Mountain ' ) on the album Finally It 's Christmas References ( edit )
Jump up ^ Marsh ,
</item 4>

<item 5>
Go Tell It on the Mountain (song)
3 AM 1965 : Ray Conniff Singers on the album Here We Come A-Caroling 1971 : Bob Marley and the Wailers featuring Peter Tosh on the album The Best of The Wailers
1981 : Anne Murray on the album Christmas Wishes 1990 : Dolly Parton on the album Home for Christmas 1993 : Bruce Cockburn on the album Christmas 1996 : Vanessa
L. Williams on the album Star Bright 1998 : Cece Winans on the album His Gift 1999 : Garth Brooks on the album Garth Brooks and the Magic of Christmas 1999 :
Jewel on the album Joy : A Holiday Collection 2001 : Candi Staton on the album Christmas in My Heart 2001 : Fred Hammond on the album Christmas ... Just
Remember 2001 : Sara Evans on the album Country Christmas 2001 2002 : Jim Nabors on the album Jim Nabors Christmas 2003 : Oh Susanna on the album Maybe This
Christmas Too ?
</item 5>

<item 6>
Go Tell It on the Mountain (song)
Yule Cat Yule Lads Zwarte Piet Gift - bringers Saint Nicholas Santa Claus List of Christmas gift - bringers by country Traditions Advent calendar Advent candle
Advent wreath Boar 's Head Feast Candle arches Cards Carols by Candlelight Cavalcade of Magi Crackers Decorations Events and celebrations Feast of the Seven
Fishes Flying Santa Google Santa Tracker Hampers Las Posadas Letters Lights Markets Meals and feasts Moravian star Nine Lessons and Carols NORAD Tracks Santa
Nutcrackers dolls Ornaments Parades list Piñatas Pyramids Räuchermann Seals Secret Santa Spanbaum Stamps Stockings Tree Twelve Days Wassailing Windows Yule Goat
Yule log By country Australia and New Zealand Denmark Germany Hawaii Hungary Iceland Indonesia Ireland Mexico Norway Philippines Poland Romania Russia Scotland
Serbia Sweden Ukraine Music Carols list Hit singles UK Hit singles US Music books Carols for Choirs The Oxford Book of Carols The New Oxford Book of Carols Piae
Cantiones Other media Films Poetry `` Old Santeclaus with Much Delight '' `` A Visit from St. Nicholas '' Television specials Yule Log In modern society Advent
Conspiracy Black Friday ( partying ) Black Friday ( shopping ) Bronner
</item 6>

<item 7>
Go Tell It on the Mountain (song)
Charles ( 1997 ) . God 's Long Summer . Princeton . p. 47 . ISBN 9780691130675 . Jump up ^ Whitburn , Joel ( 2008 ) . Hot Country Songs 1944 to 2008 . Record
Research , Inc. p. 241 . ISBN 0 - 89820 - 177 - 2 . Jump up ^ `` Firestone Your Favorite Christmas Music - Volume Three - 1964 '' . DLF Music Transfer. 2006 .
Christmas Christmas Eve Children 's Day Boxing Day Nochebuena Saint Nicholas Day St.
</item 7>

<item 8>
Go Tell It on the Mountain (song)
Unsourced material may be challenged and removed . ( December 2016 ) ( Learn how and when to remove this template message ) In 1963 , the musical team Peter ,
Paul and Mary , along with their musical director Milt Okun , adapted and rewrote `` Go Tell It on the Mountain '' as `` Tell It on the Mountain '' , their
lyrics referring specifically to : Exodus and using the phrase `` Let my people go , '' but referring implicitly to the Civil Rights struggle of the early 1960s
. This version became a moderately successful single for them ( US # 33 pop , 1964 ) . According to Religious Studies professor and Civil Rights historian
Charles Marsh , it was African American Civil Rights leader Fannie Lou Hamer who combined this song with the spiritual `` Go Down Moses , '' taking the last
line of the chorus , `` Let my people go '' and substituting it in the chorus of `` Go Tell It on the Mountain . ''
</item 8>

<item 9>
In the Pines
A rendition also appears on Long John Baldry Trio - Live ( 1999 ) . Link Wray recorded two versions titled `` Georgia Pines '' and `` In the Pines '' on his
1973 folk rock release Beans and Fatback . Gene Clark recorded the song for his 1977 album Two Sides to Every Story . Charlie Feathers recorded a version in the
1980s in Memphis . The Oak Ridge Boys recorded `` In the Pines '' on their 1983 album , Deliver . Dolly Parton 's version appears on her 1994 album Heartsongs :
Live from Home . Odetta recorded the song for her 2001 Lead Belly tribute album , Looking for a Home -- Thanks to Leadbelly . Pete Seeger 's version of `` Black
Girl '' appears on the 2002 Smithsonian Folkways re-release of recordings from the 1950s and 1960s entitled American Favorite Ballads , Vol. 1 . Josh White 's
version of `` Black Girl '' appears on his 1955 album , 25th Anniversary : The Story of John Henry , A Musical Narrative .
</item 9>

<item 10>
Go Tell It on the Mountain (song)
Marsh does not document this claim , but notes that Hamer was highly active in civil rights work beginning in the 1950s , and that the use of the Exodus story
and the singing of spirituals played a central role in her activities . Little Big Town 's 2006 non-album single version reached No. 35 on the Hot Country Songs
chart . Other artists who have recorded the song ( chiefly on either Christmas - themed music albums or collections of spirituals or folk songs ) include : 195
? : Mahalia Jackson on the album Silent Night , Holy Night 1960 : Bobby Darin on the album The 25th Day of December 1964 : Gordon MacRae on the album Firestone
Presents Your Christmas Favorites , Volume 3 1964 : Bing Crosby and Frank Sinatra on the album 12 Songs of Christmas 1964 : Simon & Garfunkel on the album
Wednesday Morning ,
</item 10>
=== END ITEM id=10acf7638d5b1943 ===

=== ITEM id=ff8deb84b777e016 ===
Gold sentences and facts:
Sentence 0: The Country Music Hall of Fame and Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
  Fact 0.0: The Country Music Hall of Fame and Museum is in Nashville, Tennessee.
  Fact 0.1: The Country Music Hall of Fame and Museum is one of the world's largest museums dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.2: The Country Music Hall of Fame and Museum is one of the world's largest research centers dedicated to the preservation and interpretation of American vernacular music.
  Fact 0.3: The Country Music Hall of Fame and Museum was chartered in 1964.
  Fact 0.4: The Country Music Hall of Fame and Museum has amassed one of the world's most extensive musical collections.
Sentence 1: Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville , Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org
  Fact 1.0: The Country Music Hall of Fame and Museum was established in 1961.
  Fact 1.1: The Country Music Hall of Fame and Museum is located at 222 5th Ave S, Nashville, Tennessee 37203.
  Fact 1.2: The type of the Country Music Hall of Fame and Museum is Hall of fame.
  Fact 1.3: The Country Music Hall of Fame and Museum had 1.1 million visitors in 2015.
  Fact 1.4: The director of the Country Music Hall of Fame and Museum is Kyle Young.
  Fact 1.5: The website of the Country Music Hall of Fame and Museum is www.countrymusichalloffame.org.

Retrieved items:
<item 1>
Nashville’s Country Music Hall of Fame and Museum preserves and interprets American country music. The nonprofit Country Music Foundation was chartered in 1964,
and the museum opened in 1967; it moved downtown in 2001 and more than doubled in size in 2014. Its exhibits, education programs, Hatch Show Print and historic
RCA Studio B complement a collection of over 2.5 million artifacts, including recordings, photographs, instruments and clothing. The Hall of Fame honors major
</item 1>

<item 2>
Country Music Hall of Fame and Museum
Current museum ( edit ) To become more accessible , in May 2001 the Country Music Hall of Fame and Museum moved to a new , 130,000 - square - foot ( 12,000 m )
facility in the heart of downtown Nashville 's arts and entertainment district . In 2014 , the museum unveiled a $100 million expansion , doubling its size to
350,000 square feet of galleries , archival storage , education classrooms , retail stores , and special event space . In the museum 's core exhibition , Sing
Me Back Home : A Journey Through Country Music , visitors are immersed in the history and sounds of country music , its origins and traditions , and the stories
and voices of many of its architects . The story is revealed through artifacts , photographs , and text panels , recorded sound , vintage video , and
interactive touchscreens . Sing Me Back Home is enhanced by numerous , rotating limited - engagement exhibits . The new ACM Gallery and the Dinah and Fred
Gretsch Family Gallery offer visitors a hands - on immersion into today 's country music with artifacts from today 's country stars and a series of technology -
enhanced activities .
</item 2>

<item 3>
Country Music Hall of Fame and Museum
Located at the head of Music Row , the museum was erected on the site of a small Nashville city park . At this point , artifacts began to be displayed and a
small library was begun in a loft above one of the museum 's galleries . Early in the 1970s the basement of the museum building was partially complete , and
library expansion began , embracing not only recordings but also books and periodicals , sheet music and songbooks , photographs , business documents , and
other materials . At the outset , CMA staff had run the museum , but by 1972 the museum ( already governed by its own independent board of directors ) acquired
its own small staff , which has steadily increased to over 150 full - time professionals . Building expansion took place in 1974 , 1977 , and 1984 to store and
display the museum 's growing collection of costumes , films , historic cars , musical instruments , and other artifacts . An education department was created
to conduct ongoing programs with Middle Tennessee schools , an oral history program was begun , and a publications department was launched to handle books , as
well as the Journal of Country Music .
</item 3>

<item 4>
Country Music Hall of Fame and Museum
Contents ( hide ) 1 History of the museum 2 Current museum 3 The Country Music Hall of Fame 4 The Museum collection 5 Architecture and design 6 Timeline 7
Senior leadership 8 See also 9 Further reading 10 References 11 External links History of the museum ( edit ) The Country Music Hall of Fame and Museum is one
of the world 's largest and most active popular music research centers and the world 's largest repository of country music artifacts . Early in the 1960s , as
the Country Music Association 's campaign to publicize country music was accelerating , CMA leaders determined that a new organization was needed to operate a
country music museum and to carry out research and education activities beyond CMA 's scope as a trade organization . Toward this end , the nonprofit Country
Music Foundation ( CMF ) was chartered by the state of Tennessee in 1964 to collect , preserve , and publicize information and artifacts relating to the history
of country music . Through CMF , industry leaders raised money to build the Country Music Hall of Fame and Museum , which opened on April 1 , 1967 .
</item 4>

<item 5>
Country Music Hall of Fame and Museum
The Country Music Hall of Fame ( edit ) Membership in the Country Music Hall of Fame , the highest honor a country music professional can receive , is extended
to performers , songwriters , broadcasters , musicians , and executives in recognition of their contributions to the development of country music . The Country
Music Hall of Fame honor was created in 1961 by the Country Music Association ( CMA ) ; the first inductees were Hank Williams , Jimmie Rodgers , and Fred Rose
. Roy Acuff , the first living artist to join the Hall of Fame , was elected in 1962 . The most recent inductees ( class of 2017 ) are Jerry Reed , Don Schlitz
and Alan Jackson . Over the Hall of Fame 's history , the number of new members inducted each year has varied from one to twelve ( no nominee was inducted in
1963 , no candidate having received sufficient votes ) . The election procedure is as follows : A small CMA nominating committee drafts slates of candidates
from each category ; categories have been defined variously over the years .
</item 5>

<item 6>
Country Music Hall of Fame and Museum
Award recipients are determined through a two - stage balloting process ; the first round of voting narrows each category to five candidates ; the second round
selects winners . The large select committee of electors that votes on Hall of Fame membership is composed of CMA members who have participated in the country
music industry for at least ten years . New Hall of Fame members receive special recognition in ceremonies at the Country Music Hall of Fame and Museum . Bas -
relief portraits cast in bronze honoring each Hall of Fame member were originally displayed at the Tennessee State Museum in downtown Nashville until the
Country Music Hall of Fame and Museum opened its own building in April 1967 ; in this barn - roofed facility at the head of Music Row , the bronze plaques
comprised a special exhibit . Today the plaques are displayed in a seventy - foot - high rotunda at the museum 's enlarged downtown Nashville facility . The
museum collection ( edit ) The museum 's archival and library storage , allows for the cultivation of the museum 's collection and includes more than 2.5
million artifacts .
</item 6>

<item 7>
Country Music Hall of Fame and Museum
In addition to the galleries , the museum has the 776 - seat CMA Theater , the Taylor Swift Education Center , and multi-purpose event rental spaces . Other
historic properties of the Country Music Hall of Fame and Museum include the letterpress operation Hatch Show Print ( located inside the museum ) and Historic
RCA Studio B ( located on Music Row ) , Nashville 's oldest surviving recording studio , where recordings by Elvis Presley , Dolly Parton , Waylon Jennings ,
and many others were made . The Country Music Hall of Fame and Museum has developed multiple platforms to make its collection accessible to a wider audience .
From weekly instrument demonstrations to its flagship songwriting program for schools , Words & Music , the museum offers an aggressive schedule of educational
programs . The museum also operates CMF Records , a Grammy - winning re-issue label ( The Complete Hank Williams and Night Train to Nashville : Music City
Rhythm & Blues , 1945 - 1970 ) ; and CMF Press , a book publishing arm that releases exhibit - related books in cooperation with Vanderbilt University Press and
other major trade publishing houses .
</item 7>

<item 8>
Country Music Hall of Fame and Museum
Nashville , Tenn. : Country Music Foundation , 1990 . N.B. : The structure as described in this document is the original one which had opened in 1967 ; the book
( amply ill . in b&w and col . ) describes the work , collections , etc. of the center 's Hall of Fame , Museum ( which includes a library ) , and recording
studio , and also gives an account of the development and various aspects of the country music industry . ISBN 0 - 915608 - 12 - X References ( edit ) ^ Jump up
to : `` About '' . Country Music Hall of Fame . Retrieved January 14 , 2016 . Jump up ^ McCall , Michael ( 2012 ) . The Encyclopedia of Country Music Second
Edition . Oxford University Press . pp. 105 -- 106 . ISBN 0195395638 . Jump up ^ `` Home '' . Studio B. Retrieved August 26 , 2017 . Jump up ^ `` Country Music
Hall of Fame promotes four to VP roles '' . Nashville Post .
</item 8>

<item 9>
Country Music Hall of Fame and Museum
Country Music Hall of Fame and museum - wikipedia Country Music Hall of Fame and museum Jump to : navigation , search This article needs additional citations
for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) (
Learn how and when to remove this template message ) Country Music Hall of Fame and Museum Established 1961 ( 1961 ) Location 222 5th Ave S Nashville ,
Tennessee 37203 Type Hall of fame Visitors 1.1 million in 2015 Director Kyle Young Website www.countrymusichalloffame.org The Country Music Hall of Fame and
Museum in Nashville , Tennessee , is one of the world 's largest museums and research centers dedicated to the preservation and interpretation of American
vernacular music . Chartered in 1964 , the museum has amassed one of the world 's most extensive musical collections .
</item 9>

<item 10>
Country Music Hall of Fame and Museum
1582778 ° N 86.7761611 ° W ﻿ / 36.1582778 ; - 86.7761611 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Country_Music_Hall_of_Fame_and_Museum&oldid=817101743 '' Categories : American country music American music awards
Music halls of fame Halls of fame in Tennessee Music museums in Tennessee Museums in Nashville , Tennessee Institutions accredited by the American Alliance of
Museums Organizations established in 1964 Museums established in 1967 1961 establishments in Tennessee Country Music Hall of Fame inductees Hidden categories :
Use mdy dates from October 2017 Articles needing additional references from May 2016 All articles needing additional references Coordinates on Wikidata Talk
Contents About Wikipedia Deutsch Español فارسی Français 한국어 Bahasa Indonesia Italiano Nederlands 日本 語 Norsk Русский Simple English Svenska 中文 Edit links This
page was last edited on 26 December 2017 , at 04 : 35 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may
apply .
</item 10>
=== END ITEM id=ff8deb84b777e016 ===

=== ITEM id=6e064fb6088416e8 ===
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
9 2003 -- 2011 volatility 3.10 2008 production dispute 3.11 2014 -- 2017 oil glut 3.12 2017 - 2018 Production cut 4 See also 5 References 6 External links
Membership ( edit ) Current member countries ( edit ) As of May 2017 , OPEC has 14 member countries : six in the Middle East ( Western Asia ) , six in Africa ,
and two in South America . According to the U.S. Energy Information Administration ( EIA ) , OPEC 's combined rate of oil production ( including gas condensate
) represented 44 percent of the world 's total in 2016 , and OPEC accounted for 73 percent of the world 's `` proven '' oil reserves , including 48 percent from
just the six Middle Eastern members : Approval of a new member country requires agreement by three - quarters of OPEC 's existing members , including all five
of the founders . In October 2015 , Sudan formally submitted an application to join , but it is not yet a member . Country Region Membership Years Population (
2016 est . )
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
OPEC - wikipedia OPEC Jump to : navigation , search Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official
language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait *
Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq
Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org Organization of the Petroleum Exporting
Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded
in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria .
</item 7>

<item 8>
OPEC
As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving
OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies . OPEC 's stated mission is ``
to coordinate and unify the petroleum policies of its member countries and ensure the stabilization of oil markets , in order to secure an efficient , economic
and regular supply of petroleum to consumers , a steady income to producers , and a fair return on capital for those investing in the petroleum industry . ''
The organization is also a significant provider of information about the international oil market . As of May 2017 , OPEC 's members are Algeria , Angola ,
Ecuador , Equatorial Guinea , Gabon , Iran , Iraq , Kuwait , Libya , Nigeria , Qatar , Saudi Arabia ( the de facto leader ) , United Arab Emirates , and
Venezuela , while Indonesia is a former member . Two - thirds of OPEC 's oil production and reserves are in its six Middle Eastern countries that surround the
oil - rich Persian Gulf .
</item 8>

<item 9>
OPEC
During 1961 -- 1975 , the five founding nations were joined by Qatar ( 1961 ) , Indonesia ( 1962 -- 2008 , rejoined 2014 - 2016 ) , Libya ( 1962 ) , United Arab
Emirates ( originally just the Emirate of Abu Dhabi , 1967 ) , Algeria ( 1969 ) , Nigeria ( 1971 ) , Ecuador ( 1973 -- 1992 , rejoined 2007 ) , and Gabon ( 1975
-- 1994 , rejoined 2016 ) . By the early 1970s , OPEC 's membership accounted for more than half of worldwide oil production . Indicating that OPEC is not
averse to further expansion , Mohammed Barkindo , OPEC 's Acting Secretary General in 2006 , urged his African neighbors Angola and Sudan to join , and Angola
did in 2007 , followed by Equatorial Guinea in 2017 . Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia , and other oil - exporting
nations have attended many OPEC meetings as observers , as an informal mechanism for coordinating policies .
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
=== END ITEM id=6e064fb6088416e8 ===

=== ITEM id=ee0a4cd8d9039925 ===
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
Pakistan’s national anthem, Qaumi Taranah (“National Anthem”), is also known as “Pāk Sarzamīn” (“The Sacred Land”). Ahmad G. Chagla composed its 80-second music
in 1949; Hafeez Jullundhri wrote the three-stanza Urdu lyrics in 1952. Officially adopted in August 1954, it was first broadcast on Radio Pakistan on 13 August.
Its Persian-influenced vocabulary makes it
</item 1>

<item 2>
Qaumi Taranah
BBC News . 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ `` Indian A capella group sings Pakistan 's national anthem on 71st Independence Day '' .
Firstpost. 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ Staff , Images ( 2017 - 08 - 12 ) . `` Indians sing the Pakistani national anthem as a gift for
our Independence Day '' . Images . Retrieved 2017 - 08 - 15 . External links ( edit ) This article contains Urdu text . Without proper rendering support , you
may see unjoined letters running left to right or other symbols instead of Urdu script . Wikimedia Commons has media related to : Qaumi Taranah ( category ) Pak
Sar Zameen Shad Bad Lyrics Pakistan : Qaumī Tarānah -- Audio of the national anthem of Pakistan , with information and lyrics Ministry of Information and
Broadcasting , Government of Pakistan . `` National Anthem '' . Archived from the original on 2007 - 10 - 26 .
</item 2>

<item 3>
Qaumi Taranah
http://mazhar.dk/film/singers/ahmadrushdi/ . Retrieved 2006 - 04 - 12 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` National Anthem of Pakistan '' . Retrieved 2006
- 04 - 12 . Jump up ^ Information Ministry , Government of Pakistan . `` Basic Facts '' . Retrieved 2006 - 04 - 12 . Jump up ^ ( 1 ) , National Anthem of
Pakistan Jump up ^ ( 2 ) , The National Anthem of Pakistan Jump up ^ ( 3 ) , Switch to Urdu from English as official language a hurdle for Pakistan Jump up ^ (
4 ) , History can not be falsely written and manufactured Jump up ^ ( 5 ) , Pakistan Basic Facts Jump up ^ Pasha , Muhammad A. English Composition ( Part II ) .
Lahore : Command Publications . Jump up ^ `` Pakistan creates new anthem record '' . The Express Tribune . 2011 - 08 - 12 . Retrieved 2012 - 01 - 01 . Jump up ^
`` Pakistan reclaims anthem singing record '' .
</item 3>

<item 4>
Qaumi Taranah
In 1950 , the impending state visit of the Shah of Iran added urgency to the matter and resulted in the Government of Pakistan asking the NAC to submit an
anthem without further delay . The NAC Chairman , then Federal Minister for Education , Fazlur Rahman , asked several poets and composers to write lyrics but
none of the submitted works were deemed suitable . The NAC also examined several different tunes and eventually selected the one presented by Ahmed G. Chagla
and submitted it for formal approval . On 21 August 1949 , the Government of Pakistan adopted Chagla 's tune for the national anthem . The anthem , without
lyrics , was performed for the first time for a foreign head of state on the state visit of the Shah of Iran to Pakistan in Karachi on 1 March 1950 by a
Pakistan Navy band . It was later played for Prime Minister Liaquat Ali Khan during his official visit to the United States on 3 May 1950 . It was played before
the NAC on 10 August 1950 . Official recognition to the national anthem , however , was not given until August 1954 .
</item 4>

<item 5>
Qaumi Taranah
Official recognition to the national anthem , however , was not given until August 1954 . The NAC distributed records of the composed tune amongst prominent
poets , who responded by writing and submitting several hundred songs for evaluation by the NAC . Eventually , the lyrics written by Hafeez Jullundhri were
approved and the new national anthem was broadcast publicly for the first time on Radio Pakistan on 13 August 1954 , sung by Hafeez Jullundhri himself .
Official approval was announced by the Ministry of Information and Broadcasting on 16 August 1954 . The composer , Ahmed G. Chagla , died in 1953 , before the
new national anthem was officially adopted . In 1955 , there was a performance of the national anthem involving 11 major singers of Pakistan , including Ahmad
Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar Wasi Ali .
</item 5>

<item 6>
Qaumi Taranah
Retrieved 1 September 2016 . Jump up ^ National Anthem. http://anisshakur.tripod.com/id129.html Jump up ^ Michael Jamieson Bristow , National-Anthems.org . ``
Forty National Anthems '' . Retrieved 2006 - 04 - 12 . ^ Jump up to : Aqeel Abbas Jafari ( 2010 ) . Pakistan Chronicle ( in Urdu ) ( 1st ed . ) . 94 / 1 , 26th
St. , Ph. 6 , D.H.A. , Karachi , Pakistan : Wirsa Publishers . p. 42 . ISBN 9789699454004 . ^ Jump up to : `` National Anthem '' . Ministry of Information and
Broadcasting , Government of Pakistan . Archived from the original on 2007 - 10 - 26 . Retrieved 2007 - 11 - 29 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` Ahmad
Rushdi '' . http://mazhar.dk/film/singers/ahmadrushdi/ .
</item 6>

<item 7>
Qaumi Taranah
Qaumi Taranah - wikipedia Qaumi Taranah Jump to : navigation , search Qaumi Taranah قومی ترانہ English : National Anthem of Pakistan National Anthem National
anthem of Pakistan Lyrics Hafeez Jullundhri , June 1952 Music Ahmad G. Chagla , 21 August 1949 Adopted 13 August 1954 Relinquished Present Audio sample Qaumi
Taranah ( Instrumental ) file help Ahmad G. Chagla composed the music of the National Anthem of Pakistan in 1949 Ahmed Rushdi recorded the National Anthem of
Pakistan in 1954 The Qaumi Taranah ( Urdu : قومی ترانہ ‬ ‎ , Qaumī Tarānah pronounced ( ˈqɔː. mi ˈt̪ə. rɑː. nɑ ) , lit .
</item 7>

<item 8>
Qaumi Taranah
rɑː. nɑ ) , lit . `` National Anthem '' ) , also known as Pāk Sarzamīn ( Urdu : پاک سرزمین ‬ ‎ , pronounced ( ˈpɑːk ˈsər. zə. miːn ) , lit . `` The Sacred Land
'' ) , is the national anthem of Pakistan . Its music was composed by Ahmad G. Chagla in 1949 , preceding the lyrics , which were written by Hafeez Jullundhri
in 1952 . It was officially adopted as Pakistan 's national anthem in August 1954 and was recorded in the same year by eleven major singers of Pakistan
including Ahmad Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar
Wasi Ali .
</item 8>

<item 9>
Qaumi Taranah
1954 -- Officially adopted as the national anthem and broadcast for the first time on Radio Pakistan on 13 August 1955 -- Sung by 11 famous Pakistani singers
including Ahmad Rushdi , Shamim Bano , Kokab Jehan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zwar Hussain , Akhtar Abbas , Ghulam Dastgir , Anwar Zaheer
and Akhtar Wassi 1996 -- Rendered in electric guitar for the first time by Pakistani rock band Junoon in their album Inquilaab 2009 -- Rendered as an acoustic
instrumental for the first time by Pakistani musician Jehangir Aziz Hayat 2011 -- On 14 August , 5,857 people gathered in a stadium in Karachi to sing the Qaumi
Taranah and set a new world record for most people gathered to sing a national anthem simultaneously . 2012 -- On 20 October , 42,813 people gathered in a
stadium in Lahore to sing the Qaumi Taranah and set a new world record for most people gathered to sing a national anthem simultaneously , which was certified
by Guinness World Records .
</item 9>

<item 10>
Qaumi Taranah
Contents ( hide ) 1 History 2 Music 3 Lyrics 4 Milestones 5 See also 6 References 7 External links History ( edit ) In early 1948 , A.R. Ghani , a Muslim from
Transvaal , South Africa , offered two prizes of five thousand rupees each for the poet and composer of a new national anthem for the newly independent state of
Pakistan . The prizes were announced through a government press advertisement published in June 1948 . In December 1948 , the Government of Pakistan established
the National Anthem Committee ( NAC ) with the task of coming up with the composition and lyrics for the official national anthem of Pakistan . The NAC was
initially chaired by the Information Secretary , Sheikh Muhammad Ikram , and its members included several politicians , poets and musicians , including Abdur
Rab Nishtar , Ahmad G. Chagla and Hafeez Jullundhri . The NAC encountered early difficulties in finding suitable music and lyrics . When President Sukarno of
Indonesia became the first foreign head of state to visit Pakistan on 30 January 1950 , there was no Pakistani national anthem to be played .
</item 10>
=== END ITEM id=ee0a4cd8d9039925 ===

=== ITEM id=f90db4148adde194 ===
Gold sentences and facts:
Sentence 0: Name Type Notes Anthropomorphism Form of personification that applies human - like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw . Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors Lear 's mental deterioration . Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . ''
  Fact 0.0: Anthropomorphism is a form of personification that applies human-like characteristics to animals and objects.
  Fact 0.1: The Adventures of Pinocchio by Carlo Collodi is an example of anthropomorphism.
  Fact 0.2: The Cheshire Cat of Alice's Adventures in Wonderland by Lewis Carroll is an example of anthropomorphism.
  Fact 0.3: Hamartia is the character flaw of an initially rich and powerful hero that leads to his tragic downfall.
  Fact 0.4: Hamartia is also referred to as the tragic flaw.
  Fact 0.5: Oedipus kills his own father because he does not understand his true parentage.
  Fact 0.6: Pathetic fallacy reflects a character's, usually the protagonist's, mood in the atmosphere or inanimate objects.
  Fact 0.7: Pathetic fallacy is related to anthropomorphism and projection.
  Fact 0.8: The storm in William Shakespeare's King Lear is an example of pathetic fallacy.
  Fact 0.9: The storm in King Lear mirrors Lear's mental deterioration.
  Fact 0.10: Personification uses comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities.
  Fact 0.11: Taken from Act I, Scene II of Romeo and Juliet: "When well-appareled April on the heel / Of limping winter treads."

Retrieved items:
<item 1>
Graph query row:
entity: Characterization
relationship: RELATED_TO
related: List of narrative techniques
context: None
fact: None
page: List of narrative techniques
</item 1>

<item 2>
List of narrative techniques
An example of this is in the first chapter of Great Expectations by Charles Dickens : `` A man who had been soaked in water , and smothered in mud , and lamed
by stones , and cut by flints , and stung by nettles , and torn by briars ; who limped , and shivered , and glared and growled ; and whose teeth chattered in
his head as he seized me by the chin '' Satire The use of humor , irony or exaggeration to criticize . An example is Network . One of the earliest examples is
Gullivers Travels , written by Jonathan Swift . The television program South Park is another . Sensory detail sight , sound , taste , touch , smell . The same
as imagery The boot was tough and sinewy between his hard - biting teeth . There was no flavor to speak of except for the blandness of all the dirt that the
boot had soaked up over the years . The only thing the boot reminded him of was the smell of a wet - dog . Understatement A diminishing or softening of a theme
or effect .
</item 2>

<item 3>
List of narrative techniques
Defamiliarization Taking an everyday object and presenting it in a way that is weirdly unfamiliar so that we see the object in a new way . Coined by the early
20th - century Russian literary critic Viktor Shklovsky in `` Art as Technique . '' In , Swift 's Gulliver 's Travels when Gulliver visits the land of the
giants and sees a giant woman 's skin he sees it is anything but smooth and beautiful when viewed up close . First - person narration A text presented from the
point of view of a character , especially the protagonist , as if the character is telling the story themselves . ( Breaking the fourth wall is an option , but
not a necessity , of this format . ) Mark Twain 's Adventures of Huckleberry Finn uses the title character as the narrator , while Sherlock Holmes is primarily
told from Watson 's perspective . Magical realism Describing events in a real - world setting but with magical trappings , often incorporating local customs and
invented beliefs . Different from urban fantasy in that the magic itself is not the focus of the story . Particularly popular with Latin American authors like
Gabriel García Márquez and Jorge Luis Borges .
</item 3>

<item 4>
List of narrative techniques
29 . ISBN 1 - 134 - 50341 - 5 . Jump up ^ Heath , Peter ( May 1994 ) , `` Reviewed work ( s ) : Story - Telling Techniques in the Arabian Nights by David
Pinault '' , International Journal of Middle East Studies , Cambridge University Press , 26 ( 2 ) : 358 -- 360 , doi : 10.1017 / s0020743800060633 Jump up ^
Heath ( 1994 ) p. 360 Jump up ^ `` Personification - Examples and Definition of Personification '' . 2013 - 06 - 27 . Retrieved 2016 - 08 - 16 . Narrative
Character Antagonist Antihero Archenemy Character arc Characterization Deuteragonist False protagonist Focal character Foil Protagonist Stock character
Supporting character Tritagonist Narrator Tragic hero Plot Action Backstory Cliché Climax Cliffhanger Conflict Deus ex machina Dialogue Dramatic structure
Exposition Eucatastrophe Foreshadowing Flashback Flashforward Frame story In medias res Pace Plot device Plot twist Poetic justice Reveal Self - fulfilling
prophecy Subplot Trope
</item 4>

<item 5>
List of narrative techniques
Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II
of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . '' References ( edit ) Jump up ^ Orehovec , Barbara ( 2003 ) .
Revisiting the Reading Workshop : A Complete Guide to Organizing and Managing an Effective Reading Workshop That Builds Independent , Strategic Readers (
illustrated ed . ) . Scholastic Inc. p. 89 . ISBN 0439444047 . Jump up ^ Demchick , Harrison ( 2013 - 09 - 26 ) . `` Techniques and Tension in Breaking Bad '' .
Huffington Post . Retrieved 2017 - 11 - 14 . Jump up ^ `` Literature Glossary - Defamiliarization '' . www.shmoop.com . Retrieved 2017 - 11 - 14 . Jump up ^ ``
Allegory Examples '' . YourDictionary . Retrieved 2017 - 11 - 14 .
</item 5>

<item 6>
List of narrative techniques
List of narrative techniques - wikipedia List of narrative techniques Jump to : navigation , search A narrative technique ( also known more narrowly for
literary fictional narratives as a literary technique , literary device , or fictional device ) is any of several specific methods the creator of a narrative
uses to convey what they want -- in other words , a strategy used in the making of a narrative to relay information to the audience and , particularly , to ``
develop '' the narrative , usually in order to make it more complete , complicated , or interesting . Literary techniques are distinguished from literary
elements , which exist inherently in works of writing . Contents ( hide ) 1 Setting 2 Plots 3 Perspective 4 Style 5 Theme 6 Character 7 References Setting (
edit ) Name Definition Example Setting The setting is both the time and geographic location within a narrative or within a work of fiction . A literary element
, the setting initiates the main backdrop and mood of a story , often referred to as the story world . The novel Ulysses by James Joyce is set in Dublin ,
Ireland , the action taking place on a single day , 16 June 1904 .
</item 6>

<item 7>
Graph query row:
entity: Characterization
relationship: PART_OF
related: Character
context: None
fact: None
page: List of narrative techniques
</item 7>

<item 8>
List of narrative techniques
Her tears were a river flowing down her cheeks . Thematic patterning Distributing recurrent thematic concepts and moralistic motifs among various incidents and
frames of a story . In a skillfully crafted tale , thematic patterning may emphasize the unifying argument or salient idea disparate events and disparate frames
have in common . Each of the chapters of Ulysses by James Joyce . Character ( edit ) Name Type Notes Anthropomorphism Form of personification that applies human
- like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis
Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw .
Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in
the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors
Lear 's mental deterioration .
</item 8>

<item 9>
List of narrative techniques
res Pace Plot device Plot twist Poetic justice Reveal Self - fulfilling prophecy Subplot Trope Kishōtenketsu Setting Backstory Utopia Dystopia Alternate history
Fictional location city country universe Theme Leitmotif Metaphor Moral Motif Irony Style Allegory Bathos Diction Figure of speech Imagery Narrative techniques
Narration Stylistic device Suspension of disbelief Symbolism Tone Mode Mood Structure Linear narrative Nonlinear narrative films television series Types of
fiction with multiple endings Form Comics Epic Fable Fabliau Fairy tale Folktale Flash fiction Legend Novella Novel Parable Play Poem Screenplay Short story
Genre Action fiction Adventure Comic Crime Docufiction Epistolary Erotic Fiction Fantasy Gothic Historical Horror Magic realism Mystery Nautical Paranoid
Philosophical Picaresque Political Psychological Romance Saga Satire Science Speculative Superhero Thriller Urban Western List of writing genres Narration First
- person Multiple narrators Stream of consciousness Stream of unconsciousness Unreliable Tense Past Present Future Related Audience Author Creative nonfiction
Fiction writing Literary theory Literary science Narratology Monomyth Rhetoric Screenwriting Storytelling Tellability Literature portal Literary composition
General topics Literature Fiction writing Writer Characterization Exposition Description Writer 's block Techniques / devices
</item 9>

<item 10>
List of narrative techniques
Author surrogate Characters which are based on authors , usually to support their personal views . Sometimes an intentionally or unintentionally idealized
version of them . A variation is the Mary Sue or Gary Stu , which primarily serves as an idealized self - insertion . Socrates in the writings of Plato . Plato
never speaks in his own voice in his dialogues . In the Second Letter , it says , `` no writing of Plato exists or ever will exist , but those now said to be
his are those of a Socrates become beautiful and new '' . Breaking the fourth wall An author or character addresses the audience directly ( also known as direct
address ) . This may acknowledge to the reader or audience that what is being presented is fiction , or may seek to extend the world of the story to provide the
illusion that they are included in it . The characters in Sesame Street often break the fourth wall when they address their viewers as part of the ongoing
storyline , which is possible because of the high level of suspension of belief afforded by its audience -- children . The American political drama show House
of Cards also uses this technique frequently to let the viewers know what the main character Frank Underwood is thinking and planning .
</item 10>
=== END ITEM id=f90db4148adde194 ===
