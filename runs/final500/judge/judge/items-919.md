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

=== ITEM id=d8ca2883c0a77510 ===
Gold sentences and facts:
Sentence 0: Daren Maxwell Kagasoff ( KA - guh - sawf ) ( born September 16 , 1987 ) is an American actor . He is known for starring as Ricky Underwood on the ABC Family teen drama series The Secret Life of the American Teenager from 2008 to 2013 .
  Fact 0.0: Daren Maxwell Kagasoff is pronounced KA-guh-sawf.
  Fact 0.1: Daren Maxwell Kagasoff was born September 16, 1987.
  Fact 0.2: Daren Maxwell Kagasoff is an American actor.
  Fact 0.3: Daren Maxwell Kagasoff is known for starring as Ricky Underwood.
  Fact 0.4: Daren Maxwell Kagasoff starred as Ricky Underwood on The Secret Life of the American Teenager.
  Fact 0.5: The Secret Life of the American Teenager is an ABC Family teen drama series.
  Fact 0.6: Daren Maxwell Kagasoff starred on The Secret Life of the American Teenager from 2008 to 2013.

Retrieved items:
<item 1>
Daren Kagasoff
Kagasoff graduated from Montclair Preparatory High School in 2005 where he played on his school 's baseball team . While attending school at San Francisco State
University , Kagasoff decided to become an actor . Career ( edit ) After deciding to return to Los Angeles , Kagasoff landed the lead role in a local theater
production of Suburbia , a play that the eponymous 1996 film was based upon . After studying acting , he began auditioning for television and film roles .
During one of his first auditions , he impressed producer Brenda Hampton so much that she hand - picked him to star as Ricky Underwood in the ABC Family series
The Secret Life of the American Teenager . The show ran for five seasons . It was announced on February 24 , 2013 that Kagasoff had been cast as Alex in the Fox
pilot Delirium , based on the novel by Lauren Oliver . However , the pilot was not picked up to series . Kagasoff co-starred as Trevor in the supernatural
horror film Ouija , directed by Stiles White and co-written by Juliet Snowden and White . The film was released in October 2014 .
</item 1>

<item 2>
Daren Kagasoff
Daren Kagasoff - wikipedia Daren Kagasoff Jump to : navigation , search Daren Kagasoff At the Twilight premiere Daren Maxwell Kagasoff ( 1987 - 09 - 16 )
September 16 , 1987 ( age 30 ) Encino , California , U.S. Occupation Actor Years active 2008 -- present Daren Maxwell Kagasoff ( KA - guh - sawf ) ( born
September 16 , 1987 ) is an American actor . He is known for starring as Ricky Underwood on the ABC Family teen drama series The Secret Life of the American
Teenager from 2008 to 2013 . Contents ( hide ) 1 Early life 2 Career 3 Filmography 4 Awards and nominations 5 References 6 External links Early Life ( edit )
Kagasoff was born in Encino , California to Elise Kagasoff ( née Levy ) and Barry Kagasoff , a diamond merchant . He is the middle of three children ; he has an
older brother , Justin , and a younger sister , Natalie .
</item 2>

<item 3>
Graph query row:
person: Daren Kagasoff
work: The Secret Life of the American Teenager
role: starred as Ricky Underwood
</item 3>

<item 4>
Daren Kagasoff
^ Jump up to : Ng , Philiana ( 3 June 2013 ) . `` ' Secret Life of the American Teenager ' : Daren Kagasoff on Series Finale ( Exclusive Video ) '' . The
Hollywood Reporter . Retrieved 24 February 2015 . Jump up ^ Hibberd , James ( 8 May 2013 ) . `` Fox rejects ' Delirium ' pilot starring Emma Roberts '' .
Entertainment Weekly . Retrieved 8 May 2013 . Jump up ^ Goldberg , Lesley ( 25 February 2013 ) . `` Fox 's ' Delirium ' Finds Co-Lead in ' Secret Life of the
American Teenager ' Star '' . The Hollywood Reporter . Retrieved 24 February 2015 . Jump up ^ The Deadline Team ( 4 December 2013 ) . `` Daren Kagasoff Joins '
Ouija ' '' . Deadline.com . Retrieved 5 February 2014 . Jump up ^ Thompson , Avery ( 12 November 2014 ) .
</item 4>

<item 5>
Krysten Ritter
The The Last International Playboy Ozzy 2008 27 Dresses Gina the goth 2008 What Happens in Vegas Kelly 2009 Glock Beretta Short film 2009 Confessions of a
Shopaholic Suze Cleath - Stuart She 's Out of My League Patty How to Make Love to a Woman Lauren 2011 Killing Bono Gloria 2011 Life Happens Kim Also writer and
co-producer 2011 Margaret Shopgirl 2012 BuzzKill Nicole 2012 Vamps Stacy Daimen 2012 Refuge Amy 2014 Listen Up Philip Melanie 2014 Veronica Mars Gia Goodman
2014 Asthma Ruby 2014 Search Party Christy 2014 Big Eyes DeAnn 2017 Hero , The The Hero Lucy Television ( edit ) Year Title Role Notes Whoopi Brynn Episode : ``
The Squatters '' One Life to Live Kay 4 episodes Law & Order Tracy Warren Episode : `` Everybody Loves Raimondo 's '' Tanner on Tanner Saleswoman 2 episodes
2005 Jonny Zero Quinn Episode : `` Pilot '' 2005 -- 2006 Veronica Mars Gia Goodman Recurring role ; 8 episodes 2006 Bedford Diaries ,
</item 5>

<item 6>
Krysten Ritter
She also played the first iteration of Allison Stark on the Fox sitcom ' Til Death ( which eventually had four actresses play that role through its run ) .
Ritter at the premiere of 27 Dresses in January 2008 Ritter continued working in film , often cast as the best friend of the lead character . In 2008 , she had
supporting roles in the romantic comedies What Happens in Vegas and 27 Dresses . She co-starred in the 2009 film Confessions of a Shopaholic as Suze , the best
friend of Isla Fisher 's character . Ritter spent three months shooting She 's Out of My League in Pittsburgh in 2008 . She played Patty , the cynical best
friend of Alice Eve 's character Molly . Ritter was cast as a young Carol Rhodes in an episode of The CW 's teen drama series Gossip Girl titled `` Valley Girls
'' , which aired May 11 , 2009 . The episode was a backdoor pilot for a proposed spin - off series of the same title , set in 1980s Los Angeles , that would
chronicle the teenage years of the character Lily van der Woodsen .
</item 6>

<item 7>
Daren Kagasoff
The film was released in October 2014 . In September 2014 , Kagasoff joined the cast of the Red Band Society , a comedy - drama series on Fox for the 2014 -- 15
American television season developed by Margaret Nagle . He had a recurring role playing Kara 's love interest , Hunter Cole . The series premiered on September
17 , 2014 and was canceled after one season . Filmography ( edit ) Television and film roles Year Title Role Notes 2008 -- 2013 Secret Life of the American
Teenager , The The Secret Life of the American Teenager Richard `` Ricky '' Underwood Main role 2014 Ouija Trevor Film 2014 Blue Daren Web series ; 2 episodes
2014 Delirium Alex Sheathes Unsold television pilot ( FOX ) 2014 Stalker Eric Bates Episode : Pilot 2014 -- 2015 Red Band Society Hunter Cole Recurring role , 7
episodes 2015 Paradise Pictures Wyatt Osborne Unsold television pilot ( USA ) 2017 S.W.A.T.
</item 7>

<item 8>
Krysten Ritter
8 episodes 2006 Bedford Diaries , The The Bedford Diaries Erin Kavenaugh 2 episodes 2006 -- 2007 Gilmore Girls Lucy Recurring role ; 8 episodes 2006 -- 2007 '
Til Death Allison Stark 5 episodes 2006 Justice Eva Episode : `` Christmas Party '' 2007 Big Day Ellen Episode : `` The Ceremony '' 2009 -- 2010 Breaking Bad
Jane Margolis Recurring role ; 9 episodes 2009 Gossip Girl Young Carol Rhodes Episode : `` Valley Girls '' Gravity Lily Champagne Main role ; 10 episodes 2011
Love Bites Cassie Episode : `` Firsts '' 2012 -- 2013 Do n't Trust the B -- -- in Apartment 23 Chloe Lead role ; 26 episodes 2013 Robot Chicken Dana Polk (
voice ) Episode : `` Immortal '' 2013 Cleveland Show , The The Cleveland Show Gina ( voice ) Episode : `` California Dreamin ' ( All the Cleves Are Brown ) ''
2013 Eric Andre Show , The The Eric Andre Show Herself Episode : `` Krysten Ritter ; Dominic Monaghan '' 2014 Blacklist ,
</item 8>

<item 9>
Daren Kagasoff
Gordon Episode : `` Imposters '' Awards and nominations ( edit ) Year Award Work Result 2009 Teen Choice Award Choice TV Breakout Star : Male The Secret Life of
the American Teenager Nominated 2009 Teen Choice Award Choice Summer TV Star : Male The Secret Life of the American Teenager Won Teen Choice Award Choice TV
Actor : Drama The Secret Life of the American Teenager Nominated Teen Choice Award Teen Choice Award The Secret Life of the American Teenager Nominated 2011
Teen Choice Award Choice TV Actor : Drama The Secret Life of the American Teenager Nominated 2012 Teen Choice Award Choice Summer TV Star : Male The Secret Life
of the American Teenager Nominated References ( edit ) Jump up ^ `` Daren Maxwell Kagasoff - California , Birth Index '' . FamilySearch . Retrieved 24 February
2015 . Jump up ^ `` Barry H Kagasoff mentioned in the record of Barry H Kagasoff and Elise Levy '' . FamilySearch . Retrieved 24 February 2015 . Jump up ^
Matsushita , Elaine ( 22 March 2009 ) .
</item 9>

<item 10>
Krysten Ritter
Ritter told Philadelphia Style that she felt her `` outgoing and bubbly and funny '' personality as a performer helped her transition into acting naturally by
letting her entertain the casting people . She won several bit film roles starting in 2001 , and then played a 1950s art history student in Julia Roberts ' Mona
Lisa Smile ( 2003 ) . In 2006 , she appeared in All This Intimacy , a two - act , Off - Broadway play by Rajiv Joseph , at the Second Stage Theatre . Ritter
signed on to star in the premiere of Zach Braff 's play , All New People , at Second Stage Theatre . Anna Camp , David Wilson Barnes and Justin Bartha
co-starred in the production under the direction of Peter DuBois . Ritter had a number of guest starring roles on television , and appeared on the second season
of Veronica Mars , playing Gia Goodman , the daughter of mayor Woody Goodman ( Steve Guttenberg ) . Ritter guest - starred on Gilmore Girls for eight episodes
from 2006 to 2007 as Rory Gilmore 's friend , Lucy .
</item 10>
=== END ITEM id=d8ca2883c0a77510 ===

=== ITEM id=60ce9c79025adcd8 ===
Gold sentences and facts:
Sentence 0: The military convention between Bulgaria and the Central Powers laid down the general plan for its campaign against the Kingdom of Serbia . It severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army , which was designated part of a combined German , Bulgarian and Austro - Hungarian force commanded by Field Marshal August von Mackensen . He had recently led the German and Austro - Hungarian armies in the highly successful and victorious Gorlice -- Tarnów Offensive of the Central Powers against the Russian army on the Eastern Front . His army group was created specifically to wage war against the Serbian army in the pre-1913 borders of the country ( `` Old Serbia '' ) , to defeat it wherever it found it and to open the land route between Hungary and Bulgaria . As commander , Mackensen acted independently and received his directives only from the German High Command . However , the field marshal 's orders to his Bulgarian forces had to be relayed to the commander of the 1st Army by the Bulgarian General Staff , which left room for the latter to intervene when needed . According to the convention , the Bulgarian commander - in - chief retained full and direct control over the Bulgarian 2nd Army and its operations in Vardar Macedonia .
  Fact 0.0: The military convention between Bulgaria and the Central Powers laid down the general plan for Bulgaria's campaign against the Kingdom of Serbia.
  Fact 0.1: The military convention severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army.
  Fact 0.2: The Bulgarian 1st Army was designated part of a combined German, Bulgarian and Austro-Hungarian force.
  Fact 0.3: The combined force was commanded by Field Marshal August von Mackensen.
  Fact 0.4: Mackensen had recently led the German and Austro-Hungarian armies in the highly successful and victorious Gorlice-Tarnów Offensive of the Central Powers against the Russian army on the Eastern Front.
  Fact 0.5: Mackensen's army group was created specifically to wage war against the Serbian army in the pre-1913 borders of the country ('Old Serbia').
  Fact 0.6: Mackensen's army group was created to defeat the Serbian army wherever it found it.
  Fact 0.7: Mackensen's army group was created to open the land route between Hungary and Bulgaria.
  Fact 0.8: As commander, Mackensen acted independently.
  Fact 0.9: Mackensen received his directives only from the German High Command.
  Fact 0.10: The field marshal's orders to his Bulgarian forces had to be relayed to the commander of the 1st Army by the Bulgarian General Staff.
  Fact 0.11: The relay of orders through the Bulgarian General Staff left room for the Bulgarian General Staff to intervene when needed.
  Fact 0.12: According to the convention, the Bulgarian commander-in-chief retained full and direct control over the Bulgarian 2nd Army.
  Fact 0.13: According to the convention, the Bulgarian commander-in-chief retained full and direct control over the 2nd Army's operations in Vardar Macedonia.

Retrieved items:
<item 1>
Bulgaria during World War I
The mood in Saint Petersburg was far less optimistic , as the Russians were aware that the price tag of Romania 's entry into the war would include Bessarabia
and also feared that its intervention would only extend the already huge Eastern Front . The initial reaction of the Central Powers to the declaration of
neutrality was similar to that of the Entente . Germany and Austria - Hungary in particular were weighing the possibility of encouraging immediate Bulgarian
intervention against Serbia , because the neutrality of Italy and Romania , countries that had been their allies prior to the outbreak of the war , was a major
defeat for German and Austrian diplomacy . Radoslavov , who was generally pro-German , engaged in talks with the German and Austro - Hungarian ambassadors as
early as July 1914 , but in the end he preferred to reaffirm Bulgaria 's neutrality . On other diplomatic fronts , the prime minister achieved greater results
with the signing of a secret treaty between Bulgaria and the Ottoman Empire on 6 August 1914 . This was a mutual defense pact that would come into effect if
either party was attacked by another Balkan power . Both countries pledged not to attack other Balkan countries without consultation with each other .
</item 1>

<item 2>
Bulgaria during World War I
The summer months of 1915 saw a decisive clash between the diplomacy of the Entente and the Central Powers . Marcel Dunan , a young French historian , reporter
for the French press and witness of the critical events , summarized the importance of this period for the entire course of the war by simply naming it the ``
Bulgarian Summer '' of 1915 . Bulgaria 's strategic geographic position and strong army now more than ever could provide a decisive advantage to the side that
managed to win its support . For the Allies , Bulgaria could provide needed support to Serbia , shore up Russia 's defenses , and effectively neutralize the
Ottoman Empire , while it could ensure the defeat of Serbia for the Central Powers , cut off Russia from its allies and open the way to Constantinople , thus
securing the continuous Ottoman war effort . Both sides had promised more or less the fulfillment of Bulgaria 's national aspirations , and the only problem
facing the Bulgarian prime minister was how to secure maximum gains in exchange for minimum commitments . During this time , many Entente and Central Powers
dignitaries were sent to Sofia in an effort to secure Bulgaria 's friendship and support .
</item 2>

<item 3>
Bulgaria during World War I
Allied representatives met with the leaders of the Bulgarian opposition parties and also provided generous financial support for opposition newspapers ; they
even attempted to bribe high - ranking government officials . Germany and Austria - Hungary were not willing to remain on the sidelines and dispatched to
Bulgaria Duke John Albert of Mecklenburg , the former ambassador to the Ottoman Empire Hans Freiherr von Wangenheim and Prince Hohenlohe , who openly declared
that after the defeat of Serbia , Bulgaria would assume hegemony of the Balkans . What seized Bulgarian interest the most was indeed the balance of military
power . The situation on the major European fronts was at that time developing markedly in favor of the Central Powers , and while the Allied operation in
Gallipoli turned into a costly stalemate , the Russians were being driven out of Galicia and Poland . Under these circumstances . the Central Powers were hoping
to secure Bulgaria at last . Still , it took Entente diplomacy more than a month to give an answer to Radoslavov 's questions and the reply proved far from
satisfactory . In reality , it hardly differed from the offer the Allies presented in May .
</item 3>

<item 4>
League of Nations
Civil War ( 1922 -- 23 ) Aspects Opposition Pacifism Anti-war movement Deployment Schlieffen Plan ( German ) Plan XVII ( French ) Warfare Military engagements
Naval warfare Convoy system Air warfare Cryptography Room 40 Horse use Poison gas Railways Strategic bombing Technology Trench warfare Total war Christmas truce
Last surviving veterans Civilian impact Atrocities Prisoners Casualties Economic history 1918 flu pandemic Destruction of Kalisz Rape of Belgium German
occupation of Belgium German occupation of Luxembourg German occupation of northeastern France Ober Ost Ottoman people Armenian Genocide Assyrian genocide
Pontic Greek genocide Urkun ( Kyrgyzstan ) Blockade of Germany Women Australia Popular culture German prisoners of war in the United States Agreements Partition
of the Ottoman Empire Sykes -- Picot Agreement Agreement of Saint - Jean - de-Maurienne French - Armenian Agreement Damascus Protocol Paris Peace Conference
Venizelos -- Tittoni agreement Treaties Treaty of Brest - Litovsk Treaty of Lausanne Treaty of London Treaty of Neuilly Treaty of St.
</item 4>

<item 5>
League of Nations
Race to the Sea First Battle of Ypres 1915 Second Battle of the Masurian Lakes Second Battle of Ypres Battle of Gallipoli Second Battle of Artois Battles of the
Isonzo Gorlice -- Tarnów Offensive Great Retreat Second Battle of Champagne Kosovo Offensive Siege of Kut Battle of Loos 1916 Erzurum Offensive Battle of Verdun
Lake Naroch Offensive Battle of Asiago Battle of Jutland Battle of the Somme first day Brusilov Offensive Baranovichi Offensive Battle of Romani Monastir
Offensive Battle of Transylvania 1917 Capture of Baghdad First Battle of Gaza Zimmermann Telegram Second Battle of Arras Second Battle of the Aisne Kerensky
Offensive Third Battle of Ypres ( Passchendaele ) Battle of Mărășești Battle of Caporetto Southern Palestine Offensive Battle of Cambrai Armistice of Erzincan
1918 Operation Faustschlag Treaty of Brest - Litovsk Spring Offensive Second Battle of the Marne Battle of Baku Hundred Days Offensive Vardar Offensive Battle
of Megiddo Third Transjordan attack Meuse - Argonne Offensive Battle of Vittorio Veneto Battle of Aleppo
</item 5>

<item 6>
Bulgaria during World War I
The end of neutrality ( edit ) As 1914 came to an end , Bulgaria remained on the sidelines of World War I. Popular opinion lacked enthusiasm for entering the
conflict and supported the country 's stance of neutrality . At this point , Prime Minister Radoslavov adopted a `` wait and see '' policy while at the same
time he successfully probed the abilities of the warring alliances to satisfy Bulgarian territorial ambitions . A final commitment could be made only when one
of the sides had gained a decisive military advantage and had firmly guaranteed the fulfillment of Bulgarian national ideals . On the battlefields far from
Bulgaria , the war had entered a long period of stalemate with no side appearing to gain the upper hand . On the Western Front in February 1915 , the French
failed to break the German lines at the First Battle of Champagne , while further attempts at the Second Battle of Artois during May also came to an
unsuccessful conclusion . The Germans had decided to focus their efforts on the Eastern Front , where they had considerable success against the Russians in the
Second Battle of the Masurian Lakes in February 1915 , but their gains were largely negated at the Siege of Przemyśl in March .
</item 6>

<item 7>
Bulgaria during World War I
In reality , it hardly differed from the offer the Allies presented in May . Once again the promises lacked a clear guarantee that Serbia would cede the desired
lands and there was not even a mention of Southern Dobrudja . In the eyes of the Bulgarians , this was a manifestation of Entente helplessness in the face of
the conflicting ambitions of its smaller Balkan allies . The diplomatic positions of the Central Powers in Sofia were strengthened immensely , forcing the
Bulgarian tsar and prime minister to assume a course towards a final alignment of the country to the side of the Central Powers . In August , a Bulgarian
military mission led by Colonel Petar Ganchev , a former military attaché in Berlin , was dispatched to Germany to work out the details for a military
convention . Almost at the same time , the Minister of War Lieutenant General Ivan Fichev resigned and was replaced as minister by the pro-German Major General
Nikola Zhekov . Radoslavov also entered into talks with the Ottoman Empire , trying to gain concessions in exchange for Bulgarian benevolent neutrality .
</item 7>

<item 8>
Bulgaria during World War I
In November , while the decisive Serbian defeat unfolded , the French attempted to exert pressure on the Bulgarian 2nd Army , but were soon forced to halt their
attempts to drive north . The forces of General Maurice Sarrail that consisted of three French and one British division dug in along an 80 - kilometer front
from the river Cherna to Lake Doiran . With the fall of Pristina , Generail Sarrail realized that the Allies could no longer help the Serbians and decided to
begin pulling back his forces to Salonika . The Bulgarian High Command shifted its focus to the Allies in Macedonia and decided that the time was right to go on
the offensive . Several days were lost , however , in scouting , and it was only on 3 December that the 2nd army commenced a general advance . Nonetheless , the
French were able to retreat in good order towards Salonika . They were soon followed by the British , who were defeated at the Battle of Kosturino . On 11
December , the Bulgarian divisions reached the Greek border , where they were ordered to halt and warned repeatedly not to cross .
</item 8>

<item 9>
Bulgaria during World War I
By the middle of December , the entire Kingdom of Serbia was occupied by the armies of the Central Powers and the Allies were pushed back to Salonika by the
Bulgarians . Within two months of its entry into the war Bulgaria , achieved its main war goal : the conquest of Vardar Macedonia . For the duration of military
operations against Serbia and the Entente in 1915 , the Bulgarian Army committed a total of around 424,375 men , while its casualties were held to around 37,000
men . By the end of 1915 , the Central Powers had established firm and unbroken control over a vast territory that stretched from the North Sea to Mesopotamia .
They also drew great political and military dividends from the defeat and occupation of Serbia . Bulgaria conquered almost all the territory it desired ,
Germany gained unrestricted access to the natural resources of Ottoman Asia , the Ottomans received much needed German matériel assistance and Austria - Hungary
secured its southern flank and could completely focus its attention on the Russian and Italian fronts .
</item 9>

<item 10>
Bulgaria during World War I
Bulgaria during World war I. - wikipedia Bulgaria during World war I. Jump to : navigation , search Bulgarian campaigns during World War I , borders including
occupied territories A German postcard commemorating the entry of Bulgaria into the war . The Kingdom of Bulgaria participated in World War I on the side of the
Central Powers from 14 October 1915 , when the country declared war on Serbia , until 30 September 1918 , when the Armistice of Thessalonica came into effect .
In the aftermath of the Balkan wars of 1912 and 1913 , Bulgaria found itself isolated on the international scene , surrounded by hostile neighbors and deprived
of the support of the Great Powers . Anti-Bulgarian sentiment grew particularly in France and Russia , whose political circles blamed the country for the
dissolution of the Balkan League , an alliance of Balkan states directed against the Ottoman Empire . The failure of Bulgarian foreign policy turned revanchism
into a focus of Bulgaria 's external relations . When the First World War started in July 1914 , Bulgaria , still recovering from the negative economic and
demographic impact of recent wars , avoided direct involvement in the new conflict by declaring neutrality .
</item 10>
=== END ITEM id=60ce9c79025adcd8 ===

=== ITEM id=c1e2925b02aa91d7 ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: Super Bowl LII followed the 2017 season.
  Fact 0.2: The Eagles took their revenge in Super Bowl LII.
  Fact 0.3: The Eagles won Super Bowl LII 41 to 33.

Retrieved items:
<item 1>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 1>

<item 2>
Graph query row:
team: Philadelphia Eagles
connection: PARTICIPATED_IN
opponent: New England Patriots
game: Super Bowl XXXIX
date: February 6, 2005
</item 2>

<item 3>
Philadelphia Eagles
1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Championship seasons in bold Links
to related articles National Football League ( 2018 ) AFC East North South West Buffalo Bills Miami Dolphins New England Patriots New York Jets Baltimore Ravens
Cincinnati Bengals Cleveland Browns Pittsburgh Steelers Houston Texans Indianapolis Colts Jacksonville Jaguars Tennessee Titans Denver Broncos Kansas City
Chiefs Los Angeles Chargers Oakland Raiders NFC East North South West Dallas Cowboys New York Giants Philadelphia Eagles Washington Redskins Chicago Bears
Detroit Lions Green Bay Packers Minnesota Vikings Atlanta Falcons Carolina Panthers New Orleans Saints Tampa Bay Buccaneers Arizona Cardinals Los Angeles Rams
San Francisco 49ers Seattle Seahawks Seasons Seasons ( by team ) Preseason Hall of Fame Game American Bowl Regular season Kickoff game Monday Night Football
International Series London Toronto Bills Series List of games played outside the U.S. Thanksgiving games Christmas games Playoffs Streaks Droughts AFC
Championship NFC Championship Super Bowl champions quarterbacks Pro Bowl History League history Executive history Championship history Timeline Defunct
franchises Franchise
</item 3>

<item 4>
Philadelphia Eagles
ESPN.com placed Eagles fans fourth in the league in its 2008 survey , citing the connection between the team 's performance and the mood of the city . The last
home game which was blacked out on television in the Philadelphia market as a result of not being sold out was against the Arizona Cardinals on Sunday ,
September 12 , 1999 , which was Andy Reid 's first home game as new head coach of the Eagles . The studies note that -- win or lose -- Eagles fans can be
counted on to pack their stadium . As of August 2008 , the team had sold out 71 consecutive games , and 70,000 were on the team 's waiting list for season
tickets . Despite finishing with a 6 -- 10 record in the 2005 season , the Eagles ranked second in the NFL in merchandise sales , and single - game tickets for
the next season were sold out minutes after phone and Internet lines opened . Eagles fans have also been known to chant the famous , `` E-A-G-L-E-S -- Eagles !
! ''
</item 4>

<item 5>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 5>

<item 6>
Super Bowl XXXIX
The Patriots , who entered the Super Bowl after compiling a 14 -- 2 regular season record , became the first team since the 1997 -- 1998 Denver Broncos to win
consecutive Super Bowls . New England also became the second team after the Dallas Cowboys to win three Super Bowls in four years . The Eagles were making their
second Super Bowl appearance after posting a 13 -- 3 regular season record . The game was close throughout , with the teams battling to a 14 -- 14 tie by the
end of the third quarter . The Patriots then scored 10 points in the 4th quarter with Corey Dillon 's 2 - yard touchdown run and Adam Vinatieri 's 22 - yard
field goal . The Eagles then cut their deficit to 24 -- 21 , with quarterback Donovan McNabb 's 30 - yard touchdown pass to receiver Greg Lewis , with 1 : 48
remaining in the game but could not sustain the comeback . Overall , New England forced four turnovers , while Patriots wide receiver Deion Branch was named
Super Bowl MVP for recording 133 receiving yards and tied the Super Bowl record with 11 catches .
</item 6>

<item 7>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
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
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 9>

<item 10>
Super Bowl XXXIX
The next year , the Patriots , having faltered down the stretch , entered those playoffs as the second seed behind the Broncos , who then defeated New England
in the AFC Title Game that year en route to a title of their own in Super Bowl 50 . Two years later , the Patriots , defending champions themselves from Super
Bowl LI , returned to repeat their title only to lose the aforementioned Super Bowl LII . Final statistics ( edit ) Sources : NFL.com Super Bowl XXXIX , Super
Bowl XXXIX Play Finder NE , Super Bowl XXXIX Play Finder Phi Statistical comparison ( edit ) New England Patriots Philadelphia Eagles First downs 21 24 First
downs rushing 6 First downs passing 14 18 First downs penalty Third down efficiency 4 / 12 9 / 16 Fourth down efficiency 0 / 0 0 / 0 Net yards rushing 112 45
Rushing attempts 28 17 Yards per rush 4.0 2.
</item 10>
=== END ITEM id=c1e2925b02aa91d7 ===

=== ITEM id=564d001f277e4515 ===
Gold sentences and facts:
Sentence 0: The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season , with the Eagles taking their revenge 41 -- 33 .
  Fact 0.0: The Eagles and the Patriots met again in Super Bowl LII.
  Fact 0.1: The Eagles and the Patriots met in Super Bowl LII following the 2017 season.
  Fact 0.2: The Eagles beat the Patriots 41 to 33 in Super Bowl LII.
  Fact 0.3: The Eagles took revenge on the Patriots in Super Bowl LII.

Retrieved items:
<item 1>
Philadelphia Eagles
As of the conclusion of the 2016 season , during the Lurie era , the Eagles are 1 -- 4 in conference championship games and 0 -- 1 in Super Bowls . On August 13
, 2009 , the Eagles signed quarterback Michael Vick . On December 6 , 2009 , Andy Reid became only the 5th coach in NFL history to win 100 or more games with a
single team in a single decade . The other four are Tom Landry , Don Shula , Tony Dungy , and Bill Belichick , all Super Bowl winners . McNabb finally had a
complete receiving corps , between first round draft pick Jeremy Maclin , DeSean Jackson 's 1,000 yard season , and Brent Celek ranking among the top 5 tight
ends in the league . Without Brian Dawkins , defensive end Trent Cole stepped up and became the dominant force on defense with 12 sacks , earning him his second
trip to the Pro Bowl and All - Pro honors .
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
As a result , the Eagles and Steelers no longer played each other every year . Currently they are scheduled to meet once every four years in the regular season
, the most recent meeting being in 2016 at Philadelphia , with the Eagles winning 34 - 3 . The Steelers have lost nine straight games on the road against the
Eagles dating back to 1966 , which was also the start of the Super Bowl era . The Eagles lead the all - time series 47 -- 28 -- 3 . Logo and uniforms See also :
Uniform ( American football ) and footnote about Eagles ' uniform numbers . For several decades , the Eagles ' colors were kelly green , silver , and white . In
1954 , the Eagles , along with the Baltimore Colts , became the second team ever in the NFL to put a logo on their helmets , with silver wings on a kelly green
helmet . In 1969 , the team wore two helmet versions : Kelly green with white wings in road games , and white with kelly green wings at home . From 1970 to ' 73
, they wore the white helmets with Kelly green wings exclusively before switching back to Kelly green helmets with silver wings .
</item 4>

<item 5>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 5>

<item 6>
Philadelphia Eagles
External links Wikimedia Commons has media related to Philadelphia Eagles . Official website Philadelphia Eagles Founded in 1933 Based and headquartered in
Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack
Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop Curse of Billy Penn The Lombardi Curse Invincible Philadelphia
Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore
Frankford Yellow Jackets Pennsylvania Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty
Bowl series Body Bag Game 4th and 26 Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh
Steelers Division championships ( 13 ) 1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships
( 4 ) 1948 1949 1960 2017 ( LII ) Retired numbers 5
</item 6>

<item 7>
Graph query row:
organization: Philadelphia Eagles
super_bowl: Super Bowl LII
super_bowl_date: 2018
</item 7>

<item 8>
Philadelphia Eagles
Championship Super Bowl champions quarterbacks Pro Bowl History League history Executive history Championship history Timeline Defunct franchises Franchise
moves and mergers Los Angeles team history Proposed stadiums 1995 -- 2016 American Football League ( 1960 -- 1969 ) Playoffs Merger NFL Championship ( 1920 --
1969 ) Playoff Bowl Records individual team Super Bowl All time win -- loss Last undefeated Tied games Canceled games Controversies Business Owners Properties
Management Council Competition Committee Collective Bargaining Agreement National Football League Players Association Lockouts Media TV NFL Network NFL RedZone
Radio NFL Films Other Officials Stadiums Chronology Awards All - Pro Hall of Fame Foreign players Player conduct Suspensions Player misconduct Combine Draft
Training camp Rivalries NFL Foundation Culture Cheerleading Color Rush Mascots Lore Nicknames Numbers Retired Sports teams based in Pennsylvania Baseball MLB
Philadelphia Phillies Pittsburgh Pirates IL Lehigh Valley IronPigs Scranton / Wilkes - Barre RailRiders EL Altoona Curve Erie SeaWolves Harrisburg Senators
Reading Fightin Phils NYPL State College Spikes Williamsport Crosscutters ALPB Lancaster Barnstormers York Revolution FL Washington Wild Things Basketball NBA
Philadelphia 76ers G League Erie BayHawks Cricket Lehigh Valley Cricket Club Philadelphia Cricket Club Football NFL Philadelphia Eagles Pittsburgh
</item 8>

<item 9>
Philadelphia Eagles
Philadelphia Eagles . September 9 , 2016 . Retrieved September 17 , 2017 . Jump up ^ `` Philadelphia Eagles Team Capsule '' ( PDF ) . 2017 Official National
Football League Record and Fact Book . National Football League . August 22 , 2017 . Retrieved September 17 , 2017 . Jump up ^ `` Top 10 NFL Rivalries Of All
Time : No. 4 Giants - Eagles '' . Sports Illustrated . December 15 , 2005 . Retrieved September 17 , 2017 . Jump up ^ Chadiha , Jeffri ( October 31 , 2007 ) .
`` Ranking the NFL 's best rivalries : Where does Colts - Pats fit ? '' . ESPN.com . Retrieved April 12 , 2008 . Jump up ^ Bryan , Dave ( September 20 , 2016 )
. `` After 8 Straight Losses , Steelers Looking For Philadelphia Freedom Sunday Against Eagles '' . SteelersDepot.com . Retrieved September 17 , 2017 .
</item 9>

<item 10>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 10>
=== END ITEM id=564d001f277e4515 ===

=== ITEM id=e846220efef20771 ===
Gold sentences and facts:
Sentence 0: The original Princess Maru incarnation of the character made her cinematic debut in the 2017 film Wonder Woman , portrayed by Spanish actress Elena Anaya .
  Fact 0.0: The original Princess Maru incarnation of the character made her cinematic debut in the 2017 film Wonder Woman.
  Fact 0.1: The original Princess Maru incarnation of the character was portrayed by Spanish actress Elena Anaya in Wonder Woman.
  Fact 0.2: Elena Anaya is a Spanish actress.
Sentence 1: Doctor Poison appears in the DC Extended Universe film Wonder Woman ( 2017 ) , portrayed by Elena Anaya . In this adaptation , her identity is Isabel Maru , a Spanish chemist recruited by General Erich Ludendorff to create chemical weapons for the German army . Due to facial disfigurement , she wears a ceramic mask over the left side of her face . It marks the character 's first live action appearance , although the setting is World War I rather than the World War II setting of the comic books . When Steve Trevor steals a book of Doctor Maru 's notes on her new chemical weapons that would render the gas masks useless against them , he crash - lands on Themyscira during his escape , which alerts Diana to the current war . She initially believes that Ares is directly responsible for the war , but later learns that all he did was provide a few ideas for the weapons to the likes of Maru while the decision to use the weapons was still the choice of the humans themselves . Ares attempts to sway Diana to his point of view by offering her a chance to kill Maru for her sins and part in the war , but Diana rejects his offer upon realizing of Trevor 's sacrifice that inspires her to recognize that she must accept humanity for its flaws rather than condemn it for not being perfect . This leads Diana to spare Maru out of mercy , who then flees .
  Fact 1.0: Doctor Poison appears in the DC Extended Universe film Wonder Woman (2017).
  Fact 1.1: Doctor Poison is portrayed by Elena Anaya in Wonder Woman (2017).
  Fact 1.2: In Wonder Woman (2017), Doctor Poison's identity is Isabel Maru.
  Fact 1.3: In Wonder Woman (2017), Isabel Maru is a Spanish chemist.
  Fact 1.4: In Wonder Woman (2017), Isabel Maru was recruited by General Erich Ludendorff.
  Fact 1.5: General Erich Ludendorff recruited Isabel Maru to create chemical weapons for the German army.
  Fact 1.6: Due to facial disfigurement, Isabel Maru wears a ceramic mask over the left side of her face.
  Fact 1.7: The Wonder Woman (2017) portrayal marks the character's first live action appearance.
  Fact 1.8: In the Wonder Woman (2017) portrayal, the setting is World War I rather than the World War II setting of the comic books.
  Fact 1.9: When Steve Trevor steals a book of Doctor Maru's notes on her new chemical weapons, he crash-lands on Themyscira during his escape.
  Fact 1.10: Doctor Maru's notes describe new chemical weapons that would render the gas masks useless against them.
  Fact 1.11: Steve Trevor's crash-landing on Themyscira alerts Diana to the current war.
  Fact 1.12: Diana initially believes that Ares is directly responsible for the war.
  Fact 1.13: Diana later learns that Ares only provided a few ideas for the weapons to the likes of Maru.
  Fact 1.14: Diana later learns that the decision to use the weapons was still the choice of the humans themselves.
  Fact 1.15: Ares attempts to sway Diana to his point of view by offering her a chance to kill Maru for her sins and part in the war.
  Fact 1.16: Diana rejects Ares's offer upon realizing Steve Trevor's sacrifice.
  Fact 1.17: Steve Trevor's sacrifice inspires Diana to recognize that she must accept humanity for its flaws rather than condemn it for not being perfect.
  Fact 1.18: Diana spares Maru out of mercy.
  Fact 1.19: Maru flees after Diana spares her.

Retrieved items:
<item 1>
Doctor Poison
and have appeared as major recurring enemies for Wonder Woman . The original Princess Maru incarnation of the character made her cinematic debut in the 2017
film Wonder Woman , portrayed by Spanish actress Elena Anaya . Contents ( hide ) 1 Publication history 2 Fictional character biography 2.1 Princess Maru 2.1. 1
Golden Age 2.1. 2 Post-Crisis 2.2 Marina Maru 2.2. 1 Post-Crisis 2.2. 2 The New 52 2.2. 3 DC Rebirth 3 Powers and abilities 4 Other versions 4.1 Wednesday
Comics 4.2 DC Super Friends 5 In other media 5.1 Television 5.2 Film 5.3 Video games 6 See also 7 References 8 External links Publication history ( edit ) The
Princess Maru incarnation of Doctor Poison first appeared in Sensation Comics # 2 and was created by William Moulton Marston and Harry G. Peter . The second
Doctor Poison first appeared in Wonder Woman Vol. 2 # 151 and was created by Eric Luke and Matthew Clark .
</item 1>

<item 2>
Doctor Poison
DC Rebirth ( edit ) After the events of DC Rebirth , Doctor Poison 's history had been altered . In the current continuity , she is Colonel Marina Maru , a
Japanese soldier working for the organization called Poison which had been founded by her family . During Wonder Woman 's first few months in the United States
, the amazon discovered a group of men infected with the Maru Virus , a poison that drives its victims to rage - induced murder . Ten years later after Wonder
Woman had discovered she had been living a lie for several years , Veronica Cale sent Doctor Poison to attack the Amazon . Powers and abilities ( edit ) Doctor
Poison is an expert in the use of poisons , toxins , and plagues . Other versions ( edit ) Wednesday Comics ( edit ) Doctor Poison appears in the Wonder Woman
sections of anthology series Wednesday Comics . DC Super Friends ( edit ) Doctor Poison is among several scientist - themed super-villains who band together to
fight the Super Friends in the story `` Weird Science '' from the series DC Super Friends .
</item 2>

<item 3>
Doctor Poison
Teen Titans Justice League Action Wonder Woman ( 2017 film ) Justice League Miscellaneous Alternative versions Earth - Two Bizarra Olive Byrne Cultural impact
Professor Marston and the Wonder Women Literature Wonder Women ! The Untold Story of American Superheroines Portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Doctor_Poison&oldid=835962156 '' Categories : Wonder Woman characters Characters created by William Moulton Marston
Characters created by H.G. Peter Comics characters introduced in 1942 Comics characters introduced in 1999 DC Comics supervillains Golden Age supervillains
Female characters in comics Female characters in film Fictional princesses Fictional mass murderers Fictional colonels Fictional female doctors Fictional
Japanese military personnel Fictional chemists Fictional pathologists Fictional toxicologists Fictional virologists Nazis in comic book fiction Hidden
categories : Pages using citations with accessdate and no URL Wikipedia articles in need of updating from June 2017 All Wikipedia articles in need of updating
Character pop Converting comics character infoboxes Talk Contents About Wikipedia Español 한국어 Simple English Tagalog Edit links This page was last edited on 11
April 2018 , at 20 : 42 .
</item 3>

<item 4>
Doctor Poison
She initially believes that Ares is directly responsible for the war , but later learns that all he did was provide a few ideas for the weapons to the likes of
Maru while the decision to use the weapons was still the choice of the humans themselves . Ares attempts to sway Diana to his point of view by offering her a
chance to kill Maru for her sins and part in the war , but Diana rejects his offer upon realizing of Trevor 's sacrifice that inspires her to recognize that she
must accept humanity for its flaws rather than condemn it for not being perfect . This leads Diana to spare Maru out of mercy , who then flees . Video games (
edit ) Doctor Poison appears as a playable character in the mobile game DC Legends . See also ( edit ) List of Wonder Woman enemies References ( edit ) Jump up
^ Sensation Comics # 2 ( 1942 ) Jump up ^ Sensation Comics # 24 ( 1943 ) Jump up ^ Wonder Woman # 28 ( 1948 ) Jump up ^ Wonder Woman : Our Worlds at War # 1
Jump up ^ Wonder Woman # 81 Jump up ^ Wonder Woman # 48 ( Mar 2016 ) Jump up ^ Wonder Woman vol.
</item 4>

<item 5>
Doctor Poison
Doctor Poison - Wikipedia Doctor Poison Jump to : navigation , search This article needs to be updated . Please update this article to reflect recent events or
newly available information . ( June 2017 ) Doctor Poison Doctor Poison debuts in Sensation Comics # 2 . Publication information Publisher DC Comics First
appearance ( Princess Maru ) Sensation Comics # 2 ( February 1942 ) ( Marina Maru ) Wonder Woman # 151 ( Vol. 2 ) ( December 1999 ) Created by William Moulton
Marston Harry G. Peter In - story information Alter ego Princess Maru Marina Maru Team affiliations Villainy Inc . Secret Society of Super-Villains Godwatch
Abilities Scientific genius in fields of chemistry and biology Engineering of pathogens , poisons , toxins , plagues and chemical - based weapons . Doctor
Poison is the name of two fictional characters , supervillains who appear in DC Comics publications and related media . Both villains were members of the
super-villain team Villainy Inc. and have appeared as major recurring enemies for Wonder Woman .
</item 5>

<item 6>
Doctor Poison
2 # 151 and was created by Eric Luke and Matthew Clark . Fictional character biography ( edit ) Princess Maru ( edit ) Golden Age ( edit ) Doctor Poison first
appeared as the chief of the poison division for a Nazi spy band who had planned to contaminate the United States Army 's water with `` Reverso , '' a drug that
compels whoever takes it to `` do the exact opposite of what they are told . '' She disguised her sex by wearing a bulky hooded costume and a mask . Doctor
Poison 's underlings captured Steve Trevor and brought him to their base in America where he was questioned by Doctor Poison . Wonder Woman , disguised as a
nurse , aided Steve Trevor but was forced to flee the scene after a lengthy battle . Meanwhile , the Reverso drug was successful in turning thousands of
American soldiers against their superiors . Wonder Woman recruited Etta Candy to help her create a diversion for Doctor Poison 's troops , which led to the
defeat of the villain . When Wonder Woman pulled her disguise off , she discovered Doctor Poison to be Maru , a Japanese princess .
</item 6>

<item 7>
Doctor Poison
( 2006 -- 07 ) Amazons Attack ! ( 2007 ) The Circle ( 2008 ) Ends of the Earth ( 2008 ) Rise of the Olympian ( 2009 ) Flashpoint ( 2011 ) The Lies ( 2016 ) Year
One ( 2016 ) The Truth ( 2017 ) Godwatch ( 2017 ) Technology Bracelets Golden Girdle of Gaea Invisible plane Lasso of Truth Mental radio Pegasi Purple Ray Sky
Kangas In other media Super Friends ( episodes ) Wonder Woman ( 1974 film ) Wonder Woman ( TV series ) ( episodes ) Justice League ( episodes ) Justice League
Unlimited ( episodes ) Justice League : The New Frontier Wonder Woman ( 2009 film ) Justice League : Crisis on Two Earths Superman / Batman : Apocalypse Wonder
Woman ( 2011 TV pilot ) Young Justice DC Universe Online Justice League : Doom Justice League : The Flashpoint Paradox Justice League : War The Lego Movie
Justice League : Throne of Atlantis Justice League : Gods and Monsters Batman v Superman : Dawn of Justice Justice League vs.
</item 7>

<item 8>
Doctor Poison
Peter Other contributors Characters Wonder Women Diana Prince Orana Artemis of Bana - Mighdall Hippolyta Donna Troy Supporting characters Antiope Aphrodite
Artemis Artemis of Bana - Mighdall Batman Drusilla Etta Candy Fury Hephaestus Hera Heracles / Hercules Hermes I Ching Julia and Vanessa Kapatelis Mala Nemesis (
Thomas Tresser ) Nubia The Olympian Orion Paula Philippus Poseidon Queen Hippolyta Helena Sandsmark Sarge Steel Superman Steve Trevor Wonder Girl ( Cassie
Sandsmark Donna Troy ) Zeus Zola Adversaries Aegeus Angle Man Ares / Mars Baron Blitzkrieg Baroness Paula von Gunther Blue Snowman Veronica Cale Captain Wonder
Cheetah Circe Dark Angel Decay Doctor Cyber Doctor Poison Doctor Psycho Duke of Deception Earl of Greed Egg Fu / Chang Tzu Eviless The First Born Genocide
Giganta Hades Hypnota Kung Lord Conquest Mask Medusa Minister Blizzard Osira Queen Clea Red Panzer Silver Swan Tezcatlipoca Zara Factions Amazons of Themyscira
Amazons of Bana - Mighdall Children of Ares Godwatch Gorilla Knights Olympian Gods
</item 8>

<item 9>
Doctor Poison
5 , # 13 ( Dec 2016 ) Jump up ^ Wonder Woman vol. 5 , # 12 ( Dec 2016 ) Jump up ^ Wednesday Comics # 2 ( 2009 ) Jump up ^ DC Super Friends # 24 ( 2010 ) Jump up
^ O'Hara , Helen ( April 2017 ) . `` Alpha Female '' . Empire . pp. 60 -- 67 . access - date = requires url = ( help ) Jump up ^
https://community.wbgames.com/t5/Update-Notes/DC-Legends-Update-1-12/td-p/1519906 Beatty , Scott ( 2009 ) . Wonder Woman : The Ultimate Guide To The Amazon
Princess . Dorling Kindersley Publishing . pp. 82 -- 83 . ISBN 0 - 7894 - 9616 - X . External links ( edit ) Doctor Poison Rapsheet Gay League Profile The
Unofficial Doctor Poison Biography ( hide ) Wonder Woman William Moulton Marston Elizabeth Holloway Marston H.G. Peter Other contributors Characters Wonder
Women Diana Prince Orana Artemis of Bana - Mighdall
</item 9>

<item 10>
Doctor Poison
In other media ( edit ) Television ( edit ) A version of Doctor Poison appears in the Batman : The Brave and the Bold episode `` Joker : The Vile and the
Villainous . '' She is seen working as a bartender at a tavern populated by D - list supervillains when Joker pays a visit to it and meets Weeper . Film ( edit
) Doctor Poison appears in the DC Extended Universe film Wonder Woman ( 2017 ) , portrayed by Elena Anaya . In this adaptation , her identity is Isabel Maru , a
Spanish chemist recruited by General Erich Ludendorff to create chemical weapons for the German army . Due to facial disfigurement , she wears a ceramic mask
over the left side of her face . It marks the character 's first live action appearance , although the setting is World War I rather than the World War II
setting of the comic books . When Steve Trevor steals a book of Doctor Maru 's notes on her new chemical weapons that would render the gas masks useless against
them , he crash - lands on Themyscira during his escape , which alerts Diana to the current war .
</item 10>
=== END ITEM id=e846220efef20771 ===

=== ITEM id=7ff97e33c2fcae84 ===
Gold sentences and facts:
Sentence 0: The first mention of the expression `` Apostles ' Creed '' occurs in a letter of AD 390 from a synod in Milan and may have been associated with the belief , widely accepted in the 4th century , that , under the inspiration of the Holy Spirit , each of the Twelve Apostles contributed an article of a creed .
  Fact 0.0: The first mention of the expression Apostles' Creed occurs in a letter of AD 390.
  Fact 0.1: The letter was from a synod in Milan.
  Fact 0.2: The expression Apostles' Creed may have been associated with a belief.
  Fact 0.3: The belief was widely accepted in the 4th century.
  Fact 0.4: The belief was that, under the inspiration of the Holy Spirit, each of the Twelve Apostles contributed an article of a creed.
Sentence 1: The title Symbolum Apostolicum ( Symbol or Creed of the Apostles ) appears for the first time in a letter , probably written by Ambrose , from a Council in Milan to Pope Siricius in about AD 390 `` Let them give credit to the Creed of the Apostles , which the Roman Church has always kept and preserved undefiled '' . But what existed at that time was not what is now known as the Apostles ' Creed but a shorter statement of belief that , for instance , did not include the phrase `` maker of heaven and earth '' , a phrase that may have been inserted only in the 7th century .
  Fact 1.0: The title Symbolum Apostolicum (Symbol or Creed of the Apostles) appears for the first time in a letter.
  Fact 1.1: The letter was probably written by Ambrose.
  Fact 1.2: The letter was from a Council in Milan to Pope Siricius.
  Fact 1.3: The letter dates to about AD 390.
  Fact 1.4: The letter says: Let them give credit to the Creed of the Apostles, which the Roman Church has always kept and preserved undefiled.
  Fact 1.5: What existed at that time was not what is now known as the Apostles' Creed.
  Fact 1.6: What existed at that time was a shorter statement of belief.
  Fact 1.7: That shorter statement did not include the phrase maker of heaven and earth, for instance.
  Fact 1.8: The phrase maker of heaven and earth may have been inserted only in the 7th century.

Retrieved items:
<item 1>
Apostles' Creed
Anglican communion ( edit ) The Apostles ' Creed is used in the non-Eucharistic services of Matins and Evening Prayer ( Evensong ) . It is invoked after the
recitation or singing of the Canticles , and it is the only part of the services in which the congregation traditionally turns to face the altar , if they are
seated transversely in the quire . Episcopal Church ( United States ) ( edit ) The Episcopal Church uses the Apostles ' Creed as a Baptismal Covenant for those
who are to receive the Rite of Baptism . Regardless of age , candidates are to be sponsored by parents and / or godparents . Youths able to understand the
significance of the Rite may go through the ritual speaking for themselves . Younger children and infants rely on their sponsors to act upon their behalf . 1 .
The celebrant calls for the candidates for Baptism to be presented . 2 . The catechumen or sponsors state their request for Baptism . 3a .
</item 1>

<item 2>
Apostles' Creed
See also ( edit ) Christianity portal Nicene Creed Chalcedonian Creed Lutheran Book of Worship Virgin birth of Jesus Passion of Jesus One , Holy , Catholic ,
and Apostolic Church Notes ( edit ) Jump up ^ Not in the sense that the word `` symbol '' has in modern English , but in the original meaning of the word ,
derived from `` Latin symbolum , sign , token , from Greek σύμβολον , token for identification ( by comparing with its counterpart ) , from συμβάλλειν , to
throw together , compare '' . Jump up ^ In a suggested revision of 1923 , this line has `` of the Holy Ghost '' . The Scottish Book of Common Prayer ( 1637 )
has `` which was conceived of the holy Ghost '' . This goes back to the 1559 Book of Common Prayer , which has `` whiche was conceived of the holy Ghost '' .
Jump up ^ This ecumenical version of the English Language Liturgical Consultation ( ELLC ) and its predecessor .
</item 2>

<item 3>
Apostles' Creed
1 Evangelical Lutheran Worship 5.4. 2 Lutheran Service Book 5.4. 3 Church of Denmark 5.5 United Methodist Church 6 Liturgical use in Western Christianity 6.1
Catholic Church 6.1. 1 Rite of Baptism 6.1. 2 Profession of faith at Mass 6.1. 3 Use in the Divine Office 6.2 Anglican Communion 6.2. 1 Episcopal Church (
United States ) 6.3 Lutheran Church 6.4 Methodism 7 See also 8 Notes 9 References 10 Further reading 11 External links 11.1 English translations Origin ( edit )
The title Symbolum Apostolicum ( Symbol or Creed of the Apostles ) appears for the first time in a letter , probably written by Ambrose , from a Council in
Milan to Pope Siricius in about AD 390 `` Let them give credit to the Creed of the Apostles , which the Roman Church has always kept and preserved undefiled ''
.
</item 3>

<item 4>
Apostles' Creed
Apostles ' Creed - wikipedia Apostles ' Creed Jump to : navigation , search The rubric above this 13th - century illuminated manuscript translates `` twelve
articles of faith set out by twelve apostles '' . The Apostles ' Creed ( Latin : Symbolum Apostolorum or Symbolum Apostolicum ) , sometimes entitled Symbol of
the Apostles , is an early statement of Christian belief -- a creed or `` symbol '' . It is widely used by a number of Christian denominations for both
liturgical and catechetical purposes , most visibly by liturgical Churches of Western tradition , including the Catholic Church , Lutheranism and Anglicanism .
It is also used by Presbyterians , Methodists and Congregationalists . The Apostles ' Creed is Trinitarian in structure with sections affirming belief in God
the Father , Jesus Christ His Son and the Holy Spirit . The Apostles ' Creed was based on Christian theological understanding of the Canonical gospels , the
letters of the New Testament and to a lesser extent the Old Testament . Its basis appears to be the old Roman Creed known also as the Old Roman Symbol .
</item 4>

<item 5>
Apostles' Creed
Its basis appears to be the old Roman Creed known also as the Old Roman Symbol . Because of the early origin of its original form , it does not address some
Christological issues defined in the Nicene and other Christian Creeds . It thus says nothing explicitly about the divinity of either Jesus or the Holy Spirit .
This makes it acceptable to many Arians and Unitarians . Nor does it address many other theological questions which became objects of dispute centuries later .
The first mention of the expression `` Apostles ' Creed '' occurs in a letter of AD 390 from a synod in Milan and may have been associated with the belief ,
widely accepted in the 4th century , that , under the inspiration of the Holy Spirit , each of the Twelve Apostles contributed an article of a creed . Contents
( hide ) 1 Origin 2 Musical settings 3 Text in Latin 4 Text in Greek 5 English translations 5.1 Ecumenical version 5.2 Roman Catholic Church 5.2. 1 Articles 1
-- 12 5.2. 2 Articles 9 -- 10 5.3 Church of England 5.4 Lutheran Church 5.4.
</item 5>

<item 6>
Apostles' Creed
180 ) . While the individual statements of belief that are included in the Apostles ' Creed -- even those not found in the Old Roman Symbol -- are found in
various writings by Irenaeus , Tertullian , Novatian , Marcellus , Rufinus , Ambrose , Augustine , Nicetas , and Eusebius Gallus , the earliest appearance of
what we know as the Apostles ' Creed was in the De singulis libris canonicis scarapsus ( `` Excerpt from Individual Canonical Books '' ) of St. Pirminius (
Migne , Patrologia Latina 89 , 1029 ff . ) , written between 710 and 714 . Bettenson and Maunder state that it is first from Dicta Abbatis Pirminii de singulis
libris canonicis scarapsus ( idem quod excarpsus , excerpt ) , c. 750 . This longer Creed seems to have arisen in what is now France and Spain . Charlemagne
imposed it throughout his dominions , and it was finally accepted in Rome , where the Old Roman Symbol or similar formulas had survived for centuries .
</item 6>

<item 7>
Apostles' Creed
Ἀμήν . English translations ( edit ) Ecumenical version ( edit ) 15th - century Flemish tapestry illustrating the first four articles of the Creed The English
Language Liturgical Consultation ( ELLC ) is an international ecumenical group whose primary purpose is to provide ecumenically accepted texts for those who use
English in their liturgy . In 1988 it produced a translation of the Apostles ' Creed , distinguished among other things by its avoidance of the word `` his ''
in relation to God . The text is as follows : I believe in God , the Father almighty , creator of heaven and earth . I believe in Jesus Christ , God 's only Son
, our Lord , who was conceived by the Holy Spirit , born of the Virgin Mary , suffered under Pontius Pilate , was crucified , died , and was buried ; he
descended to the dead . On the third day he rose again ; he ascended into heaven , he is seated at the right hand of the Father , and he will come to judge the
living and the dead .
</item 7>

<item 8>
Apostles' Creed
But what existed at that time was not what is now known as the Apostles ' Creed but a shorter statement of belief that , for instance , did not include the
phrase `` maker of heaven and earth '' , a phrase that may have been inserted only in the 7th century . This illumination from a 13th - century manuscript shows
the apostles writing the Creed , receiving inspiration from the Holy Spirit . The account of the origin of this creed , the forerunner and principal source of
the Apostles ' Creed , as having been jointly created by the Apostles under the inspiration of the Holy Spirit , with each of the twelve contributing one of the
articles , was already current at that time . The earlier text evolved from simpler texts based on Matthew 28 : 19 , part of the Great Commission , and it has
been argued that it was already in written form by the late 2nd century ( c. 180 ) .
</item 8>

<item 9>
Apostles' Creed
It has been argued nonetheless that it dates from the second half of the 5th century , though no earlier . Some have suggested that the Apostles ' Creed was
spliced together with phrases from the New Testament . For instance , the phrase `` descendit ad inferos '' ( `` he descended into hell '' ) echoes Ephesians 4
: 9 , `` κατέβη εἰς τὰ κατώτερα μέρη τῆς γῆς '' ( `` he descended into the lower , earthly regions '' ) . It is of interest that this phrase first appeared in
one of the two versions of Rufinus in AD 390 and then did not appear again in any version of the creed until AD 650 . This phrase and that on the communion of
saints are articles found in the Apostles ' Creed , but not in the Old Roman Symbol nor in the Nicene Creed . Musical settings ( edit ) Musical settings of the
Symbolum Apostolorum as a motet are rare .
</item 9>

<item 10>
Apostles' Creed
Archived from the original on June 5 , 2011 . Retrieved May 19 , 2011 . ^ Jump up to : `` Apostles ' Creed '' , Dictionary of the Christian Church , Oxford
University Press , 2005 , p. 90 , ISBN 978 - 0 - 19 - 280290 - 3 . Jump up ^ Day , Gardiner Mumford ( 1963 ) , The Apostles ' Creed : an interpretation for
today , Scribner , p. 33 . Jump up ^ McGiffert , Arthur Cushman ( 2008 ) , The Apostles ' Creed : Its Origin , Its Purpose , and Its Historical Interpretation ,
p. 42 , ISBN 0 - 559 - 85199 - 5 . Jump up ^ Bettenson , Henry , ed. ( 1963 ) , Documents of the Christian Church ( 2nd ed . ) , London , p. 23 . Jump up ^
Lynch , Joseph ( 1992 ) , The Medieval Church , London and NY : Longman , p. 7 .
</item 10>
=== END ITEM id=7ff97e33c2fcae84 ===
