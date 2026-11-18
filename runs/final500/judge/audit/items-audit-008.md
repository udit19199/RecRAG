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

This file has 20 items.

=== ITEM id=a182-dc64e3234bd1b571 ===
Gold sentences and facts:
Sentence 0: Barbara Eden as Stella Johnson , a widowed cosmetics salesperson with a young daughter , whose joie de vivre , her principles , and her individuality anger most of the powerful people in town and on the Parent Teachers Association board . She inevitably becomes PTA president after embarrassing Flora , the snobby former President and her equally snobby friends . Ronny Cox as Willis `` Will '' Newton , a wealthy member of the PTA who does n't agree with Flora and her snobby allies and is in love with Stella . He and Stella get married at the end of the movie . Nanette Fabray as Alice Finley , Stella 's best friend and co-conspirator against the snobs of Harper Valley . She owns and operates the La Moderne Beauty Shop . Louis Nye as Kirby Baker , a local real estate agent who is allied with Flora and her friends . Notorious for being a lecherous playboy and infamous for impregnating his secretaries , one of whom , Gladys Wilmot , left town to see an abortionist in Cincinnati . As part of a plan to discredit Stella , he tries to foreclose on Stella 's house , which infuriates her . He then gets arrested and jailed for attacking Myrna Wong , a friend of Stella 's , who rebuffs his advances , throwing him by use of martial arts and destroying the office . John Fiedler as Bobby Taylor , a man on the PTA who lusts after Stella . Overly dominated by his wife , Holly . Although open to allow Stella to speak , he is not above having a fellow PTA board member kidnapped to commit election fraud . He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others ) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main antagonist . She is the pompous and extremely snobbish president of the PTA board who thoroughly despises Stella ( who defeats her for president at the end of the movie ) and all she stands for . So desperate is she to retain her power and prestige in the community that she has a fellow member of the PTA ( Skeeter Duggan ) kidnapped so she and her allies can commit election fraud . She is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon . Molly Dodd as Olive Glover , the recording secretary of the PTA who is a gambling addict . Clearly the most corrupt and most vindictive member of the PTA board , she is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her . She is the one who initially angers Stella by sending her the letter denouncing her , and tries to get Dee 's friend , Mavis , arrested for embezzlement . Fay DeWitt as Willamae Jones , a snide biology teacher at the junior high school who is also a member of the PTA . She makes no secret of the fact that she favors the Reilly twins , due to the connections between her and Flora . Notorious for being a nymphomaniac ( though she hides it under a prim - and - proper façade ) , she is caught in flagrante delicto with Barney Crunk , the town 's mailman , on film . DeVera Marcus as Holly Taylor , Bobby 's obviously overbearing wife , who tends to drink a lot when Bobby is out of town , and does n't approve of Bobby 's lusting , or of anything else he does ( this is different from the character in the song , who `` seems to use a lot of ice whenever ( her husband ) 's away '' , implying adultery with the iceman ) . However , despite her ties to Flora and her scheming friends , she , like Shirley Thompson , is spared any retribution from Stella , due to Stella refusing to date Bobby . Louise Foley as Mavis , Dee Johnson 's best friend , who is nearly framed for a crime she did not commit by the vindictive Olive Glover . Brian Cook as Carlyle , a popular track star who chooses Dee , after she gets a makeover , and angers Bettina Reilly in the process . Laura Teige as Bettina Reilly , One of Flora 's twin granddaughters , who wants to be with Carlyle , and is jealous when he chooses Dee over her . She tends to be as snotty as her grandmother . Jan Teige as Edwina Reilly , Bettina 's twin sister , who is just as snotty as her . Irene Yah - Ling Sun as Myrna Wong , a friend of Stella 's who is a judo expert ; she is assaulted by Kirby Baker ( after leading him on ; she destroys him and his office in retaliation ) and has him sent to jail for assault . Tobias Anderson as `` Cranky '' Barney Crunk , the town 's nosy mailman who had the bad habit of peeking in everyone 's mail . He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson - Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her house is destroyed by pink - painted elephants .
  Fact 0.18: Alice Finley operates the La Moderne Beauty Shop.

Retrieved items:
<item 1>
Harper Valley PTA (film)
She is terrified when her house is destroyed by pink - painted elephants . Production and release ( edit ) Filming ( edit ) Harper Valley PTA was filmed in
twenty - seven days from October 3 , 1977 to December 8 , 1977 . It was filmed on location for one week in the town of Lebanon , Ohio and then continued in Los
Angeles , California . The track meet scene was filmed at Simi Valley High School On October 31 , 1977 , production ceased temporarily when Nanette Fabray was
knocked to the ground by a spooked elephant while shooting a scene involving pink elephants in the Hancock Park neighborhood of Los Angeles , California ; she
suffered a severe concussion , bruises and back sprains and was taken to Cedars - Sinai Medical Center where her condition was reported as serious but stable .
Following her recovery , filming resumed on November 30 . Release ( edit ) Harper Valley PTA opened in six theaters in Lebanon , Cincinnati and Dayton , Ohio on
May 23 , 1978 in accordance with executive producer Phil Borack 's plan to release the film first in smaller markets , where regional success could encourage
exhibitors in bigger cities to book the film .
</item 1>

<item 2>
Harper Valley PTA (film)
He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally
against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the
local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion
from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge
schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family
of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a
stampede of elephants , painted pink , stormed through his house , destroying it .
</item 2>

<item 3>
Harper Valley PTA (film)
The film starred Barbara Eden , Nanette Fabray , Ronny Cox , Louis Nye and Susan Swift , directed by Richard Bennett , and was mainly released to drive - in
theaters throughout the summer of 1978 . The film has a cult following in fans of the original song . The film 's promotional tagline is : `` The song was
scandalous . The movie is hilarious ! '' Contents ( hide ) 1 Plot 2 Cast and characters 3 Production and release 3.1 Filming 3.2 Release 4 Soundtrack 5
Television series 6 References 7 External links Plot ( edit ) Stella Johnson is a beautiful widowed single mother who lives in the town of Harper Valley , Ohio
. She sells cosmetics door - to - door for the fictitious AngelGlo Cosmetics and is not afraid to enjoy life . Her fourteen - year - old daughter , Dee , is a
student at Harper Valley Junior High School . After leaving school , Dee brings her mother a letter from the school 's Parent Teacher Association board , which
is led by the pompous and snobbish Flora Simpson - Reilly .
</item 3>

<item 4>
Harper Valley PTA (film)
The film grossed over $2 million its opening weekend , $5.8 million in its first three weeks and over $16 million after twelve weeks . The film opened in Los
Angeles on August 2 , 1978 and in New York City on January 12 , 1979 . Soundtrack ( edit ) Harper Valley PTA : Original Soundtrack Recording Soundtrack album by
various artists Released June 1978 Recorded Singleton Sound Studios Nashville , Tennessee Genre Soundtrack Length 29 : 50 Label Plantation Records Producer
Shelby S. Singleton , Jr . Singles from Harper Valley PTA `` Harper Valley PTA ( re-issue ) '' Released : July 1978 In June 1978 , a soundtrack album was
released on vinyl , cassette and 8 - track tape by Plantation Records . To promote the film 's release and its soundtrack , the title song by Jeannie C. Riley
was re-issued as a single . The album made its compact disc premiere through Varèse Sarabande on October 27 , 2017 .
</item 4>

<item 5>
Harper Valley PTA (film)
He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson -
Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could
commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry
Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to
his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her
house is destroyed by pink - painted elephants .
</item 5>

<item 6>
Harper Valley PTA (film)
Harper Valley PTA ( film ) - wikipedia Harper Valley PTA ( film ) Jump to : navigation , search Harper Valley PTA Theatrical release poster Directed by Richard
Bennett Produced by George Edwards Written by George Edwards Barry Schneider Starring Barbara Eden Nanette Fabray Ronny Cox Louis Nye Susan Swift Pat Paulsen
Music by Nelson Riddle Cinematography Willy Kurant Edited by Michael Economou Distributed by April Fools Productions Release date May 23 , 1978 ( 1978 - 05 - 23
) ( Midwest ) June 2 , 1978 ( 1978 - 06 - 02 ) ( Canada ) August 2 , 1978 ( 1978 - 08 - 02 ) ( Los Angeles ) January 12 , 1979 ( 1979 - 01 - 12 ) ( New York
City ) Running time 93 minutes Country United States Language English Budget $1 million Box office $25 million ( USA ) Harper Valley PTA is a 1978 American
comedy film inspired by the popular 1968 country song `` Harper Valley PTA '' written by Tom T. Hall and performed by country singer Jeannie C. Riley .
</item 6>

<item 7>
Harper Valley PTA (film)
Barbara Eden reprised her role as Stella Johnson for the series which lasted two seasons and a total of 30 episodes were produced . References ( edit ) Jump up
^ St. Petersburg Times Jump up ^ Milwaukee Sentinel : Elephant injures Nanette Fabray Jump up ^ Observer Reporter External links ( edit ) Harper Valley PTA on
IMDb Harper Valley PTA at the TCM Movie Database Harper Valley PTA at AllMovie Harper Valley PTA at the American Film Institute Catalog Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Harper_Valley_PTA_(film)&oldid=818590841 '' Categories : 1978 films English - language films 1970s comedy films
American comedy films American films American independent films Feminist films Films about revenge Films adapted into television programs Films based on songs
Films set in Ohio Films shot in Ohio Films shot in Los Angeles 1978 soundtracks Film soundtracks Hidden categories : Articles with hAudio microformats Album
infoboxes lacking a cover Music infoboxes with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 4 January 2018 , at 12
: 58 .
</item 7>

<item 8>
Harper Valley PTA (film)
Side 1 `` Harper Valley PTA '' -- performed by Jeannie C. Riley `` Dee 's Visit '' ( instrumental ) -- performed by Nelson Riddle `` Mr. Harper '' -- performed
by Barbara Eden `` Alice 's Place '' ( instrumental ) -- performed by Nelson Riddle `` High School Confidential '' -- performed by Jerry Lee Lewis `` Willie Mae
'' ( instrumental ) -- performed by Nelson Riddle Side 2 `` Harper Valley PTA '' ( instrumental ) -- performed by Nelson Riddle `` Widow Jones '' -- performed
by Barbara Eden `` Twin Tune '' ( instrumental ) -- performed by Nelson Riddle `` Ballad of a Teenage Queen '' -- performed by Johnny Cash `` Ice Cream Disco ''
( instrumental ) -- performed by Nelson Riddle `` Whatever Happened to Charlie Brown '' -- performed by Rita Remington & Carol Channing `` Ending ( Reprise :
Harper Valley PTA ) '' ( instrumental ) -- performed by Nelson Riddle Television series ( edit ) Main article : Harper Valley PTA ( TV series ) In 1981 , Harper
Valley PTA was made into a television sitcom ( created by Sherwood Schwartz ) which aired on NBC from January 1981 to May 1982 .
</item 8>

<item 9>
Harper Valley PTA (film)
Also shown are Edwina , Bettina 's identical twin sister who is just as snobbish as Bettina , and Dee 's best friend Mavis . Real estate agent Kirby Baker , a
member of the PTA Board , plans to ruin Stella by foreclosing on her house ( which his company owns ) , but is arrested for assaulting Myrna Wong , an Asian -
American martial arts expert helping Stella to set up the lecherous Baker . Now things become more and more desperate . The board finally decide to resort to
criminal means to maintain the power they hold , which is fast slipping away thanks to Stella 's growing popularity . The board members then decide to hire a
couple of kidnappers named Dutch and Tex to have Skeeter abducted so they can commit election fraud . The kidnappers snatch Skeeter as he is taking the trash
out and take him in a nearby abbey , where he is imprisoned and made drunk on wine .
</item 9>

<item 10>
Harper Valley PTA (film)
He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who
tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others
) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being
manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she
was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is
sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main
antagonist .
</item 10>
=== END ITEM id=a182-dc64e3234bd1b571 ===

=== ITEM id=a179-88b2d38dcf4ec12c ===
Gold sentences and facts:
Sentence 0: The Raiders had a great chance to score a touchdown early in the game after cornerback Charles Woodson intercepted Buccaneers quarterback Brad Johnson 's pass on the third play of the game and returned it 12 yards to the Tampa Bay 36 - yard line . However , six plays later , Tampa Bay defensive end Simeon Rice sacked Raiders quarterback Rich Gannon on third down , forcing Oakland to settle for kicker Sebastian Janikowski 's 40 - yard field goal to give them a 3 -- 0 lead .
  Fact 0.6: Kicker Sebastian Janikowski kicked a 40-yard field goal for Oakland.

Retrieved items:
<item 1>
Super Bowl XXXVII
The Raiders came into the game as four - point favorites . However , the Tampa Bay defense dominated the contest . Oakland quarterback Rich Gannon threw a Super
Bowl record five interceptions , three of which were returned for touchdowns . The Buccaneers also sacked Gannon five times , and scored 34 consecutive points
to build a 34 -- 3 lead late in the third quarter . Tampa Bay safety Dexter Jackson , who had two of those interceptions and returned them for 34 yards , was
named Super Bowl MVP. Jackson became only the second safety and third defensive back named Super Bowl MVP .
</item 1>

<item 2>
Super Bowl XXXVII
But that changed when the team hired Tony Dungy as head coach in 1996 . Dungy , along with his defensive coordinator Monte Kiffin , rebuilt the defense around a
core group of players , such as defensive lineman Warren Sapp , linebacker Derrick Brooks , and defensive backs Ronde Barber and John Lynch . By 1997 , Tampa
Bay ranked 3rd in the league in total yards allowed , 2nd in 1998 , and 3rd in 2000 . With one of the best defenses in the league , Dungy helped guide the
Buccaneers to four playoff appearances in his six years as the team 's head coach . But the team always had one of the worst offenses in the league , and this
was a major factor in their playoff losses . Thus , Dungy was fired and replaced by Gruden . Still , even Gruden had trouble getting the offense in sync during
his first year as Tampa Bay 's head coach . In 2002 , the Buccaneers ranked 25th in the league in total yards gained ( 5,222 ) .
</item 2>

<item 3>
Super Bowl XXXVII
Then on third and 18 from the Oakland 29 - yard line , Tampa Bay linebacker Derrick Brooks intercepted a pass from Gannon and returned it 44 yards for a
touchdown , giving the Buccaneers a 41 -- 21 lead with only 1 : 18 left , and leading Buccaneers radio announcer Gene Deckerhoff to make his famous call of ``
The dagger 's in , we 're gon na win the Super Bowl ! '' . A few plays later , with the Raiders now playing for pride , Dwight Smith intercepted a tipped pass
and returned it 50 yards for a touchdown with only two seconds remaining . Gramatica kicked the extra point , and the Buccaneers were up 48 -- 21 . Raiders
defensive tackle Chris Cooper returned Gramatica 's kickoff 6 yards before being tackled by Jack Golden , ending the game . The Buccaneers won their first -
ever Super Bowl with a 48 - 21 victory . Gannon said after the game that his performance was `` nightmarish '' .
</item 3>

<item 4>
Super Bowl XXXVII
January 15 , 2003 . Retrieved August 27 , 2012 . Jump up ^ Mitchell , Fred ( January 27 , 2003 ) . `` In their fashion , Bucs fans laugh '' . Chicago Tribune .
Retrieved August 27 , 2012 . Jump up ^ Vivinetto , Gina ( January 27 , 2003 ) . `` Super Bowl XXXVII : Superstars belt out favorites , or lip synch them '' .
St. Petersburg Times . ^ Jump up to : George , Thomas ( January 27 , 2003 ) . `` Bucs steal the treasure '' . The New York Times . Retrieved August 27 , 2012 .
^ Jump up to : `` Tampa Bay Buccaneers 48 vs. Oakland Raiders 21 '' . pro-football - reference . Retrieved August 27 , 2012 . Jump up ^ Cunningham , Michael ;
Skolnick , Ethan J. ( January 27 , 2003 ) . `` Treu Gets The Call In Robbins ' Absence '' . South Florida Sun - Sentinel .
</item 4>

<item 5>
Super Bowl XXXVII
Better still for the Buccaneers was that Oakland had n't changed their audible - calling signals that Gruden himself had installed , thus tipping off plays
repeatedly . Johnson finished the game with 18 out of 34 completions for 215 yards and 2 touchdowns , with 1 interception , along with 10 rushing yards .
Pittman was the top rusher of the game with 129 yards . Alstott was the game 's second leading rusher with 15 yards and a touchdown , and had 5 receptions for
43 yards . Wide receiver Keyshawn Johnson was the Buccaneers ' leading receiver with 6 catches for 69 yards . Smith recorded 2 interceptions , 94 return yards ,
and 2 touchdowns . He also added another 23 yards on a kickoff return . Gannon finished the game 24 out of 44 for 272 yards and 2 touchdowns , but was
intercepted a Super Bowl record 5 times . Garner was their leading rusher , but with only 10 yards , and caught 7 passes for 51 yards . Rice was the Raiders '
leading receiver of the game with 5 catches for 77 yards and a touchdown .
</item 5>

<item 6>
Super Bowl XXXVII
Quarterback Brad Johnson made the Pro Bowl , completing 281 out of 451 passes for 3,049 yards , 22 touchdowns , and only 6 interceptions . Running back Michael
Pittman led the team in rushing with 718 yards and one touchdown , and caught 59 passes for 477 yards . Pro Bowl fullback Mike Alstott had 548 rushing yards and
5 touchdowns , and also had 35 receptions for 242 yards and 2 touchdowns . Wide receiver Keyshawn Johnson led the team with 76 receptions for 1,088 yards and 5
touchdowns , while wide receiver Keenan McCardell had 61 receptions for 670 yards and 6 touchdowns . However , the Buccaneers ' defense was still the strength
of the team , leading the NFL in total defense ( 252.8 yards per game ) , pass defense ( 155.6 yards per game ) , points allowed ( 12.3 points per game ) ,
passing touchdowns allowed ( 10 ) , interceptions ( 31 ) , and opponent passer rating ( 48.4 ) . Brooks , Lynch , Sapp , and defensive end Simeon Rice all had
Pro Bowl years .
</item 6>

<item 7>
Super Bowl XXXVII
Gurley 91 Chuck Darby 92 Anthony McFarland 93 DeVone Claybrooks 94 Greg Spires 95 Ron Warner 96 Ellis Wyms 97 Simeon Rice 98 Corey Smith 99 Warren Sapp -- Onome
Ojo -- T.T. Toliver Head coach : Jon Gruden Coaches : Joe Barry Jeremy Bates Richard Bisaccia Michael Christianson Jay Gruden Monte Kiffin Richard Mann Rod
Marinelli Raheem Morris Bill Muir Johnny Parker Stan Parrish Mike Tomlin Art Valero Kirby Wilson Tampa Bay Buccaneers Founded in 1976 Based and headquartered in
Tampa , Florida Franchise Franchise Seasons History Coaches Players Expansion draft First - round draft picks Draft history Starting quarterbacks Broadcasters
Stadiums Tampa Stadium Raymond James Stadium Culture and lore Hugh Culverhouse Malcolm Glazer Tampa 2 Captain Fear Cheerleaders Snow Bowl A Triumph of the Heart
: The Ricky Bell Story Rivalries Carolina Panthers New Orleans Saints Miami Dolphins Playoff appearances ( 10 ) 1979 1981 1982 1997 1999 2000 2001 2002 2005
2007 Division championships ( 6 ) 1979 1981 1999
</item 7>

<item 8>
Super Bowl XXXVII
Defensive back Tory James was also a big contributor with 4 interceptions . Playoffs ( edit ) Further information : 2002 -- 03 NFL playoffs The second - seeded
Buccaneers defeated the fourth - seeded San Francisco 49ers , 31 -- 6 , and the top - seeded Philadelphia Eagles on the road , 27 -- 10 , in the playoffs , to
make the Super Bowl for the first time in franchise history . Meanwhile , the top - seeded Raiders were victorious against the fourth - seeded New York Jets ,
30 -- 10 , and the second - seeded Tennessee Titans , 41 -- 24 . The Raiders won against the Titans through Gannon 's pass - oriented offense . Super Bowl
pre-game News ( edit ) The Raiders entered the game favored to win in their first Super Bowl in 19 years . They were also the first franchise to appear in the
Super Bowl in four different decades ( 1960s , 1970s , 1980s , 2000s ; the 1990s was the only decade that they did n't appear in the Super Bowl ) . However ,
much of the media hype surrounded the Gruden trade prior to the season .
</item 8>

<item 9>
Super Bowl XXXVII
This was the last Super Bowl until Super Bowl 50 to take place in California . It was the first and only time until Super Bowl XLVIII in Super Bowl history that
the league 's number one - ranked offense ( Raiders ) was pitted against the league 's number one - ranked defense ( Buccaneers ) . Super Bowl XXXVII is also
referred to as the `` Pirate Bowl '' , due to both teams ' pirate - themed names . The game is sometimes referred to as the `` Gruden Bowl '' , because the
primary storyline surrounding the game revolved around Jon Gruden . Gruden was the Raiders ' head coach from 1998 to 2001 , and then became the Buccaneers coach
in 2002 . Tampa Bay , `` Gruden 's new team '' , made their first Super Bowl appearance in team history after posting a 12 -- 4 regular season record . Oakland
, `` Gruden 's old team '' , advanced to their fifth Super Bowl after an 11 -- 5 regular season . This was also the first Super Bowl to feature two former
division rivals , as the Buccaneers and Raiders both played in the AFC West in 1976 .
</item 9>

<item 10>
Super Bowl XXXVII
Aided with Pittman 's gains of 6 and 19 yards , the Buccaneers scored their first touchdown on a 2 - yard run from fullback Mike Alstott , increasing their lead
to 13 -- 3 . Then with 3 : 45 left in the half , Tampa Bay drove 77 yards , assisted by a pair of catches by Alstott for 28 total yards . Johnson finished the
drive with a 5 - yard touchdown pass to wide receiver Keenan McCardell to give the Buccaneers a 20 -- 3 halftime lead . Third quarter ( edit ) Tampa Bay
continued to dominate the game for most of the third quarter . The Buccaneers forced the Raiders to punt on the opening drive of the second half . Next , Tampa
Bay marched 89 yards on a 14 - play drive that took 7 : 52 off the clock , and ended with Johnson 's 8 - yard touchdown pass to McCardell to increase their lead
to 27 -- 3 .
</item 10>
=== END ITEM id=a179-88b2d38dcf4ec12c ===

=== ITEM id=a155-e36d289150b52f23 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Italian Title / RAI English Title 4Kids English title Italian air date American air date `` Una Fata a Gardenia '' / `` An Unexpected Event '' `` It Feels Like Magic '' 28 January 2004 ( 2004 - 01 - 28 ) 19 June 2004 While on summer vacation , Bloom , an ordinary 16 - year - old high school girl living on Earth , encounters a battle between a 17 - year - old fairy named Stella and an ogre named Knut . As Stella 's magic scepter is stolen , Bloom intervenes , discovering that she has magical powers and manages to turn the tide . Stella invites Bloom to attend Alfea , a school for fairies , to fully master her new powers . Knut returns with more ghouls and a troll , launching a surprise attack at Bloom 's home . Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky , Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School '' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents , she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy , Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her . Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school , Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded . She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball . Later Bloom spots the Trix trying to steal Stella 's ring , and tries to fight them off , transforming in a fairy for the first time . Although the Trix are able to escape , the other Winx girls inform Bloom that Stella 's ring is safe and that they swapped the contents of the case with a baby duck ( Pepe ) that bonds with Icy . `` La palude di Melmamora '' / `` The Black - Mud Swamp '' `` The Voice of Nature '' 4 February 2004 ( 2004 - 02 - 04 ) 10 July 2004 As part of Professor Palladium 's field trip , the class must navigate the Black - Mud Swamp to reach a clearing in the Gloomy - Wood Forest by listening to nature and without using any magic . The boys from Red Fountain are transporting a troll prisoner but the Trix cause their aircraft to crash land , and the troll is freed . The Winx and the Red Fountain boys end up meeting in the swamp , and must work together , despite their differences , to find their way out . They encounter the troll who is harassing some other Alfea fairies , and chase him off , however , the Trix find the troll and send him away . Although the Red Fountain boys failed to deliver the troll , they remark they are starting to feel like a team . The Winx girls , although they finished last in their assignment , are commended by their teachers for their helping others . 5 5 `` Appuntamento al buio '' / `` Date with Disaster '' `` Date with Disaster '' 6 February 2004 ( 2004 - 02 - 06 ) 17 July 2004 While the Winx are on kitchen duty , Stella is invited on a date with Prince Sky ( Brandon ) at the brand - new Black Lagoon restaurant and also asks for help on picking out an outfit . While selecting her attire , Stella decides not to wear the ring of Solaria and gives it to Bloom for safekeeping ( believing she wo n't need her magic powers ) . The next morning , Stella starts acting completely different , becoming irritable and refusing to say antything about last night . During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria ! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower '' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out what made the Trix go to such lengths to get it . Meanwhile , the Trix use a spell on the ring , hoping to access an extremely powerful magic known as the Dragon Flame , only to have their spell blow up in their faces as the ring proves negative ! An infuriated Icy throws the ring away and plans to begin the search all over again . Back at Alfea , Bloom learns more on the ring of Solaria , but also learns about the Dragon Flame . Bloom then turns to Faragonda for answers , who tells her that the Dragon Flame was the energy of the Great Dragon , who created the magical universe and later rested on the planet Domino . That night , the Winx plan to infiltrate Cloud Tower to retrieve Stella 's ring ( since she ca n't transform without it ) . Using the same secret tunnel the Trix used earlier , the girls sneak into Cloudtower without a hitch while the Trix are out . After searching their room , Stella finds her ring , but the tunnel does n't reopen and the Winx are forced to find the main exit without getting caught . While exploring , the Winx stumble upon Cloudtower 's archive which contains history on every person in the Magic dimension . A curious Bloom spots a book with her name on it and tries to read more on her past , but quickly regrets it when it turns into a trap that alerts Headmistress Griffin ! The Winx try to make a hasty escape , but Griffin sends Bako monsters to scare them out ! After using their magic to fight the monsters , the Winx are led to a dead end . In a panic to destroy the last monster , Stella accidentally sets the room on fire ! As the Winx begin to suffocate , a mysterious voice guides Bloom to an exit , allowing them to leave Cloudtower . Though the ring was recovered , the Winx are busted since Griffin already called Faragonda hours ago . As punishment , Faragonda temporarily extracts all their powers . 7 7 `` A che servono gli amici ? '' / `` Friends in Need '' `` Grounded '' 11 February 2004 ( 2004 - 02 - 11 ) 31 July 2004 The Winx have lost all their magic as punishment for sneaking into Cloudtower , but to make matters worse , Griselda tasks them with cleaning the entire school without their magic ( or technology , much to Tecna 's dismay ) while the rest of the students and faculty go to a concert . Problems continue as Tecna struggles with no gadgets to help her , causing her to wear a bucket on her head and attempt to use a broom , while Musa , Stella and Bloom have a water fight , wasting their time , energy and supplies . In order to make the punishment lighter , the girls secretly invite the boys from Redfountain over after everyone has left , claiming to be throwing an all - night party . Meanwhile , the Trix use the opportunity to sneak into a near - empty Alfea to find the real source of the Dragon Flame . The boys arrive and instead learn they 'll be working as janitors , though they still play music as they work . Soon after , the Trix teleport to Alfea and summon the vacuums to guide them towards the Dragon Flame , while distracting the gang by summoning a monster . The Winx and specialists quickly meet their distraction , and while the boys try to fight with their weapons , their quickly knocked aside . The Winx , despite having no magic , manage to outsmart the beast by using floor cleaner . Unable to find the Dragon Flame , the Trix depart to Cloudtower , unseen by anyone while the Specialists head back to Redfountain . The next morning , Faragonda sees the damage and hears the story from the Winx and as a reward for defeating an adversary using only wits and resourcefulness , the Winx 's powers are returned to them . 8 8 `` La festa della rosa '' / `` A Friendship Sundered '' `` The Day of the Rose '' 13 February 2004 ( 2004 - 02 - 13 ) 7 August 2004 During the Rose Festival , both Bloom and Riven unwittingly fall into a trap by the Trix to divide the close relationship between the Specialists and the Winx . In the end , Riven thinks that Darcy has saved him in a racing crash and he blames Bloom for what happened then gets mad at his friends for believing Bloom over him and transfers out of the squad of Specialists along with Darcy . 9 9 `` Il tradimento di Riven '' / `` Betrayed ! '' `` Spelled '' 16 February 2004 ( 2004 - 02 - 16 ) 14 August 2004 There are hostilities between the Winx due to Stella 's behavior , and Musa 's feelings for Riven are strained . In a fight with the Trix , the possessor of the Dragon 's Flame is revealed to be Bloom . 10 10 `` La fiamma del drago '' / `` Bloom Tested '' `` Magical Reality Check '' 18 February 2004 ( 2004 - 02 - 18 ) 21 August 2004 It 's time for Bloom 's magical midterm test in a virtual reality machine . The Trix mess with the computer so they can enter it and test her to be sure of Bloom is indeed the one who possesses the Dragon Flame . 11 11 `` Il regno delle ninfe '' / `` The Monster and The Willow '' `` Junior League '' 20 February 2004 ( 2004 - 02 - 20 ) 28 August 2004 Flora experiments with her plants to study for her test . Stella ( who had her clothes torn apart by a plant ) and Tecna ( who was attacked by a plant ) tell Flora to move the plants . She moves them to Black - Mud Swamp . Flora seeks help from the Water Nymphs in finding a special plant . The Water Nymphs tell the Winx about a monster which has been terrorizing them and the Winx decide to fight it . 12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom , which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13 13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop . A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 - 02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March 2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03 ) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17 `` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! '' / `` The Font of Dragon Fire '' `` Senior Witches Go to Earth '' 8 March 2004 ( 2004 - 03 - 08 ) 16 October 2004 Upset about the events at Redfountain , Bloom returns to Earth . The Trix take advantage of the situation and attack her on Earth . The witches reveal to Bloom that she is the surviving heir to planet Domino ( Sparks ) , and they steal the Dragon Fire from her . 19 19 `` Attacco ad Alfea '' / `` The Fall of Magix '' `` The Army of Decay '' 10 March 2004 ( 2004 - 03 - 10 ) 23 October 2004 Bloom returns to Alfea following the events on Earth . The Trix take control of Cloud Tower and summon the Army of Decay to take control of Magix . 20 20 `` La scomparsa di Bloom '' / `` Mission to Domino '' `` Sparks of Hope '' 12 March 2004 ( 2004 - 03 - 12 ) 30 October 2004 To counter the powerful magic possessed by the Trix , the Winx Club go to planet Domino hoping to find the Dragon 's Flame . Redfountain is attacked by the Army of Decay . 21 21 `` Trappola di ghiaccio '' / `` The Crown of Dreams '' `` The Frozen Palace '' 15 March 2004 ( 2004 - 03 - 15 ) 6 November 2004 In the ruins of Domino ( Sparks ) , the Winx are attacked during their exploration . In Magix , Redfountain is destroyed by the Trix . 22 22 `` Il ritorno di Riven '' / `` Storming Cloudtower '' `` Mission to Cloud Tower '' 17 March 2004 ( 2004 - 03 - 17 ) 13 November 2004 The Army of Decay forces the retreat of the population to Alfea . The Winx go to Cloud Tower in attempt to retrieve the power of the Dragon 's Flame from the Trix . Riven rejoins the Specialists . 23 23 `` Fuga da Torrenuvola '' / `` Power Play '' `` The Search for the Flame '' 19 March 2004 ( 2004 - 03 - 19 ) 20 November 2004 The Trix torment the Winx for being unable to find the power of the Dragon 's Flame on Domino ( Sparks ) . The cadre and staff of Cloud Tower escape to Alfea . 24 24 `` Il mistero del lago '' / `` The Witches ' Siege '' `` Battle for Alfea '' 22 March 2004 ( 2004 - 03 - 22 ) 27 November 2004 The Army of Darkness launches an attack on Alfea . Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice . 25 25 `` Il sonno di Magix '' / `` The Ultimate Challenge '' `` The Great Witch Invasion '' 24 March 2004 ( 2004 - 03 - 24 ) 4 December 2004 The first attack on Alfea fails . Bloom follows the spirit of her older - sister Daphne and regains her Dragon Fire powers . The Trix lead the Army of Decay in a final attack on Alfea . 26 26 `` Battaglia finale '' / `` The Witches ' Downfall '' `` Fire and Ice '' 26 March 2004 ( 2004 - 03 - 26 ) 11 December 2004 As the Army of Darkness approaching Alfea and the Trix seeking to take complete control of the Magix Dimension , the Winx , except Bloom and including Alfea 's students and teachers take action to fend off the Trix from completely destroying algae to the ground before the Trix use the powers of the Dragon . RC1 RC1 `` Attack of the Senior Witches '' N / A 22 May 2004 A clip show episode made by 4Kids Entertainment featuring scenes from episodes 18 , 19 and 20 where the Trix attack Alfea , serving as a sneak preview to the first season .
  Fact 0.121: Bloom's curiosity about her past and magical powers begins to interfere with her confidence.

Retrieved items:
<item 1>
Bloom’s attempt to investigate her past at Cloudtower triggers a trap, but a mysterious voice helps the Winx escape. Faragonda punishes them by removing their
powers. While powerless, they clean Alfea, outwit a monster with floor cleaner, and earn their magic back; meanwhile, the Trix search for the Dragon Flame. The
Trix later manipulate Bloom and Riven, straining friendships and driving Riven from the Specialists, though he eventually returns. Bloom is revealed as the
</item 1>

<item 2>
Winx Club (season 1)
Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head
Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello
di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school ,
Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded
. She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix
casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball .
</item 2>

<item 3>
Bloom reveals her real name to Faragonda, who allows her to stay; the girls name their group the Winx. The Trix repeatedly sabotage school life and test Bloom’s
connection to the Dragon Flame. Bloom discovers Mike found her as a baby during a fire, learns she is Domino’s surviving heir, and seeks answers at Cloud Tower.
She also learns Sky and Brandon have switched identities. The Trix steal Bloom’s Dragon Flame, seize Cloud Tower, and unleash the Army of Decay.
</item 3>

<item 4>
Winx Club (season 1)
A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning
Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 -
02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower
with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March
2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test .
</item 4>

<item 5>
Winx Club (season 1)
Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky ,
Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School
'' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her
admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents ,
she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy ,
Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her .
</item 5>

<item 6>
Bloom joins Alfea under a false identity, befriends Musa, Tecna, and Flora, and discovers the Trix—Cloud Tower witches Icy, Darcy, and Stormy—are behind Knut
and the troll. She first transforms while protecting Stella’s Ring of Solaria. The Winx and Red Fountain Specialists cooperate to escape a swamp and confront
the troll. The Trix trick and kidnap Stella to steal her ring; Bloom gives it up to save her, emerging as
</item 6>

<item 7>
Winx Club (season 1)
During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella
begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince
Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the
Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives
them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She
quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls
transform , and quickly learn that the entire date was a plot by the Trix .
</item 7>

<item 8>
Winx Club (season 1)
The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria
! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic
energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms
out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower
'' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out
what made the Trix go to such lengths to get it .
</item 8>

<item 9>
Winx Club (season 1)
12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud
Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other
contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom
, which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13
13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth
during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop .
</item 9>

<item 10>
Winx Club (season 1)
Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03
) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17
`` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at
Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon
switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! ''
</item 10>
=== END ITEM id=a155-e36d289150b52f23 ===

=== ITEM id=a177-7a4a864abdfa1f7e ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him . `` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied about Robin Hood 's soul having been destroyed . Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental . Belle lives on Hook 's ship ; in a deal with Gold , the Evil Queen will not harm her or the baby . Emma tells Archie about her visions and notes that Regina is n't present in them and could be the hooded figure . 114 `` The Other Shoe '' Steve Pearlman Jane Espenson & Jerome Schwartz October 9 , 2016 ( 2016 - 10 - 09 ) 4.11 In the past in the Enchanted Forest , Cinderella meets Prince Thomas at the ball , but flees when she believes he loves her stepsister Clorinda ; Clorinda actually plans to elope with Thomas ' footman , Jacob , but her mother , Lady Tremaine , abducts her to the Land of Untold Stories after forcing Cinderella to betray Clorinda 's location . In Storybrooke , Ashley hopes to reconcile with Clorinda . Aided by the Evil Queen , Tremaine lures Ashley into a trap for her and Jacob . Ashley protects Jacob ; Clorinda reunites with him and forgives Cinderella , while Tremaine is arrested . Emma struggles to control her magic . Whale helps Jekyll recreate his serum , while the Evil Queen and Hyde form an alliance and she releases him . Snow White wants normality , but David secretly investigates his father 's death ; Gold provides information in exchange for David delivering a message to Belle . 115 `` Strange Case '' Alrick Riley David H. Goodman & Nelson Soler October 16 , 2016 ( 2016 - 10 - 16 ) 3.53 In the past in Victorian England , Jekyll petitions the Academy of Science for membership but Dr. Lydgate rejects him . Rumplestiltskin arrives and magically perfects Jekyll 's serum ; Jekyll transforms into Hyde , who blackmails Lydgate for Jekyll 's membership . Mary is Lydgate 's daughter and Jekyll 's unrequited love ; she and Hyde fall in love . Jekyll accidentally fatally defenestrates Mary ; he frames Hyde . In Storybrooke , Gold imprisons Belle on the ship to protect her from Hyde ; but it is Jekyll who tries to kill Belle in revenge for Rumplestiltskin 's role in Mary 's death . Hook saves Belle by killing Jekyll , and Hyde dies of identical wounds ; Regina asks Emma to kill her if necessary to stop the Evil Queen . Gold warns Belle that their child will need his protection . Mary Margaret resumes teaching , and her assistant Shirin inspires her to draw on her life as Snow . Shirin is secretly Princess Jasmine , working with the Oracle to search for Aladdin . 116 5 `` Street Rats '' Norman Buckley Edward Kitsis & Adam Horowitz October 23 , 2016 ( 2016 - 10 - 23 ) 3.40 In the past in Agrabah , Aladdin seeks a magical weapon for Jasmine , to save the kingdom from Jafar ; he is the weapon and a Savior . Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death . Aladdin rejects Jafar and saves Jasmine . In Storybrooke , the Oracle is murdered . Emma helps Jasmine , hoping Aladdin 's survival ensures her own . Having faked his death , Aladdin reveals himself to Emma and confesses that he doomed Agrabah by using the shears , which he gives to her ; she asks Hook to dispose of them , but he secretly keeps them . The Evil Queen impersonates Archie to learn of Emma 's visions , and convinces Zelena to turn Archie back into a cricket . 117 6 `` Dark Waters '' Robert Duncan Andrew Chambliss & Brigitte Hales October 30 , 2016 ( 2016 - 10 - 30 ) 3.06 During the Dark Curse , Captain Nemo abducts Hook and invites him to join the crew `` family '' of his submarine , Nautilus , in their quest for the `` Mysterious Island '' , which is the Land of Untold Stories . Hook discovers the first mate is his half - brother , Liam , and decides to leave to avoid confrontation ; but Liam attacks him , instead mortally wounding Nemo , who steps in . In the present , Archie is freed . Henry confronts Hook over lying about the shears , and Liam imprisons them on Nautilus . Hook opens up and Henry accepts him as family ; they save each other and dispose of the shears . Modern medicine saves Nemo ; he , Liam , and Hook make amends . Emma and Aladdin discuss being a Savior . Jasmine reveals that Agrabah vanished . Snow supports Belle at her ultrasound . The Evil Queen seduces Gold and offers the retrieved shears in exchange for help taking Snow 's heart . 118 7 `` Heartless '' Ralph Hemecker Jane Espenson November 6 , 2016 ( 2016 - 11 - 06 ) 3.56 In the past , fugitive Snow and shepherd David separately travel to Longbourn ; she to flee the kingdom and he to sell his struggling farm . Without ever seeing each other 's faces , David saves Snow from a bounty hunter and she repays him with money to save the farm , deciding not to flee . Their hands touch and a magical sapling is created . In the present , the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water unless Mary Margaret and David hand over their hearts . They learn the sapling could stop the Queen , who destroys it just after they find it . But it has shown them their previous meeting , and they conclude that their love can endure anything ; they surrender themselves . The Queen casts a sleeping curse on their hearts ; because they share Snow 's heart , the curse affects them both ; each can wake the other but immediately succumbs to the curse , so they can never be awake together . Regina uses Gold 's relationship with the Queen to separate the latter from Zelena , who earns Gold 's enmity by exposing his schemes to Belle . 119 8 `` I 'll Be Your Mirror '' Jennifer Lynch Jerome Schwartz & Leah Fong November 13 , 2016 ( 2016 - 11 - 13 ) 3.40 Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror , but she traps them there , where the Dragon is already imprisoned . Henry prepares for a date with Violet ; the Queen impersonates Regina and tries to manipulate him but he recognizes her . Emma and Regina discover Sidney was building a mirror portal out of the world and try and finish it , but the Evil Queen forces the Dragon to attack Emma and Regina , using his heart , and tries to make Henry kill the Dragon with the Hammer of Hephaestus ; he instead uses it to free his mothers . Mary Margaret and David adjust to living in shifts . Belle asks Zelena to help her escape Storybrooke via magical portal , and they form an alliance with Aladdin and Jasmine . Aladdin steals the Genie 's lamp and the Sorcerer 's wand from Gold 's shop , but Gold takes the wand back from Zelena and puts a magical tracking device on Belle . He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H. Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina continue to feud . Emma obtains the sword from her vision . 121 10 `` Wish You Were Here '' Ron Underwood Edward Kitsis & Adam Horowitz December 4 , 2016 ( 2016 - 12 - 04 ) 3.27 The Black Fairy abducts Gideon . Emma discovers the sword can kill the Evil Queen without hurting Regina . The Queen steals the lamp and wishes Emma to an alternate reality where she was never the Savior ; Regina follows her . David and his allies take the lamp from the Queen , who is turned into a serpent by the hooded figure , revealed to be the adult Gideon . Jasmine wishes herself and Aladdin to Agrabah . In the `` wish realm '' , Emma lives as a princess until Regina traumatically restores her memory . Regina frees Rumplestiltskin in exchange for a magic bean , but the portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood . 122 11 `` Tougher Than the Rest '' Billy Gierhart Edward Kitsis & Adam Horowitz March 5 , 2017 ( 2017 - 03 - 05 ) 3.03 In 1990 Minnesota , a young Emma is told by a teenager that she has the power to change her fate . In present day , after they miss their way out of the Wish realm , Emma and Regina find another option for escape when they come across August , who agrees to help create a new portal . Regina goes to find Robin Hood to see if he would 've been better off without her . After Regina finds Robin , they 're caught by the Sheriff of Nottingham and later by the Wish realm Rumple . Robin believes Regina when she tells him about their being in an alternate reality . Emma discovers that August was the teenager she met back in Minnesota and inspires him to complete the portal . Emma , Regina , and Robin return to Storybrooke , where Gideon explains to Gold and Belle his intention to kill Emma so that he can acquire her savior powers and kill the Black Fairy , a move that has David furious and Belle concerned . Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12 ) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried about whether David will give him his blessing to marry Emma , but David does . Meanwhile , Regina tries to get the Wish Realm Robin assimilated into Storybrooke , but soon questions why he agreed to follow her , as it turns out that he had something else planned after he steals a chest from the vault . The truth as to who really murdered Robert is revealed when August gives Hook missing pages he took from Henry 's book . When Hook sees an illustration of David 's dad , he recognizes him and recalls murdering him during a robbery in the Enchanted Forest so that there was no witness to his crime . 124 13 `` Ill - Boding Patterns '' Ron Underwood Andrew Chambliss & Dana Horgan March 19 , 2017 ( 2017 - 03 - 19 ) 2.71 During the First Ogres War , Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old . After every last soldier had been slain , Beowulf surrenders to death just as Rumplestiltskin intervenes , killing all the ogres . Hailed as a hero in the village , Beowulf is irked by Rumple 's new title . He creates a ruse to lead Rumple to his demise . Fearing he would not be able to follow through on protecting the villagers without Dark Magic , Rumple bestows his dagger upon Baelfire as an insurance policy . Following a turbulent exchange and threat to frame Rumplestiltskin as a murderer , Beowulf retreats . Feeling the pressure of his threat , Bae commands his father to kill him . Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence . In Storybrooke , Rumple recognizes a familiar pattern in Gideon , who makes strides in his quest to kill Emma Swan at the cost of The Blue Fairy 's vitality . Hook worries about the repercussions of telling Emma his newly unveiled secret . Meanwhile , Emma unknowingly forces a proposal after happening upon Hook 's engagement ring . Robin of Locksley attempts to flee the town , but to no avail . The Evil Queen escapes her reptilian confinement and indoctrinates Robin . 125 14 `` Page 23 '' Kate Woods David H. Goodman & Brigitte Hales March 26 , 2017 ( 2017 - 03 - 26 ) 2.85 In the Enchanted Forest , The Evil Queen is in search of a spell that will help her end Snow White for good , but her father has love in mind for his daughter and leads her on a quest to Cupid 's Arrow . However , The Evil Queen rebrands the arrow to follow hatred and discovers that the person she despises the most is herself . In Storybrooke , The Evil Queen and Regina face off in a final confrontation . Just when Regina wins the upper hand , she remembers the Evil Queen 's self - loating and evenly splits the light and the darkness in their hearts , to share their love and hate and allow the Evil Queen to reform . After Emma calls off the engagement , Hook decides to board the Nautilus with Captain Nemo . Hook later changes his mind following a conversation with Snow but , before he can return , Gideon hijacks the vessel and sends it out of Storybrooke . 126 15 `` A Wondrous Place '' Steve Pearlman Jane Espenson & Jerome Schwartz April 2 , 2017 ( 2017 - 04 - 02 ) 2.80 At a loss for hope , Regina suggests a `` girls ' night out '' for Emma with her and Snow . While at the bar , she confides in the mixologist and sheds a tear . Meanwhile , the Nautilus is transported to the Enchanted Forest , where Aladdin and Jasmine board . They have yet to locate Agrabah and are growing weary . After Hook and Jasmine confide in one another , the vessel is irrevocably damaged and Jasmine uses a genie wish to relocate the group to the island where Jafar is located . They discover a cabana belonging to Ariel and Prince Eric . Ariel has collected a genie lamp that houses Jafar . Jasmine is able to sway him into divulging the location of the hidden Agrabah before defeating him with magic dust . Through true love 's kiss , Aladdin and Jasmine , along with Hook and Ariel , are transported to the restored city and Aladdin 's genie curse is broken . Ariel supplies Hook with a means to reach Emma . Emma receives the message and Gideon , no longer masquerading as the mixologist , threatens that he will use her tears to keep Hook from returning to Storybrooke unless she helps him kill the Black Fairy . Flashbacks to Ariel and Jasmine 's first meeting show the princess falter under Jafar 's trickery . 127 16 `` Mother 's Little Helper '' Billy Gierhart Story by : Edward Kitsis & Adam Horowitz Teleplay by : Paul Karp April 9 , 2017 ( 2017 - 04 - 09 ) 2.60 In Storybrooke , Gold reveals to Emma and Snow that the Black Fairy is his mother and that the Dark Curse Regina cast was created by the Black Fairy . Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is controlling him to kill Emma . 128 17 `` Awake '' Sharat Raju Andrew Chambliss & Leah Fong April 16 , 2017 ( 2017 - 04 - 16 ) 2.51 Regina works feverishly to wake Snow and Charming from the Sleeping Curse . Gold comes face to face with the Black Fairy , the powerful enemy Emma must fight in the Final Battle . Hook , still in Neverland , meets up with Tiger Lily , who gives him a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy . Gold confronts the Black Fairy over her manipulating Gideon with his heart and she maintains that Storybrooke would be destroyed if the two of them fought . A pixie flower allows Emma to open a doorway to Neverland and rescue Hook , who then re-proposes properly . The townsfolk gather around Emma 's sleeping parents and each take small amounts of the Sleeping Curse unto themselves , diluting it and allowing Snow and David to finally reunite . In 1993 , a pixie flower awakens Snow and Charming from their cursed selves , but an awakened Rumplestiltskin tells them they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior . 129 18 `` Where Bluebirds Fly '' Michael Schultz David H. Goodman & Brigitte Hales April 23 , 2017 ( 2017 - 04 - 23 ) 2.69 In Zelena 's past , her childhood friend Stanum asks for her help in finding the Crimson Heart , a magical artifact that would be able to restore his human form after having been cursed by the Wicked Witch of the North to slowly turn into tin . Zelena 's refusal to give up her magic leads to her betraying Stanum , as she keeps the Heart and leaves him to become the Tin Man . In the Present , The Black Fairy tempts Zelena with an offer to join her , only to have Zelena oppose her instead . Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm . In the Present , as everyone looks for the missing wand , Gold brings Gideon and Emma into the Dream World to search for Gideon 's heart , where the events of the flashbacks unfold . Gold then confronts Fiona , but she reveals the real reason behind why he was abandoned . Gold later retrieves Gideon 's heart and tells everyone that the Black Fairy is dead , paving the way for Emma and Hook 's wedding . However , Gold and Fiona make a deal that will see the Final Battle go forward , with Emma being the Savior that will fight in it . Meanwhile , Regina helps Zelena adjust to life without her powers . 131 20 `` The Song in Your Heart '' Ron Underwood David H. Goodman & Andrew Chambliss May 7 , 2017 ( 2017 - 05 - 07 ) 2.87 In the past , Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength , which also spreads to Oz as well . When Regina becomes affected by the wish , she finds a way to end it before the first curse takes place , but the Blue Fairy , who granted Snow the wish , tells Snow and David that the wish will be part of Emma 's destiny . In Emma 's past , her chance to make use of her talent is held back by being told she 'll be alone . This would come back to taunt Emma in the present day , as Fiona returns to use that weakness to take Emma 's heart before unleashing a new curse on her wedding day , with help from Gold by freezing her family . When Henry discovers the page in the book that reveals the song in Emma 's heart will make her stronger , not only does it embolden Emma , it stops Fiona from crushing her heart , but Fiona still tells Emma that the battle will continue . On her wedding day , Emma and Hook finally marry , just in time for the dark clouds of the Black Fairy 's curse to erupt from the clock tower and engulf them at 6 : 00 PM , setting up the final battle . 132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle . Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their `` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity . Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
  Fact 0.338: Gold retrieves Gideon's heart.

Retrieved items:
<item 1>
Once Upon a Time (season 6)
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her . Production ( edit ) Development ( edit ) Executive producers Adam
Horowitz and Edward Kitsis announced that they were ending the half - season arc structure that was seen in seasons three through five , with Horowitz saying ,
`` We 're also planning a 22 - episode story as opposed to breaking it up into two halves this year . It has been really exciting and fun . '' Kitsis added : ``
We are changing around what we 're doing this year and going back to that season 1 mentality of small town stories and smaller arcs . '' This season will focus
on Storybrooke as the main setting , but will also show new realms and there will be the exploration of the Savior 's mythology . Kitsis and Horowitz hired two
directors from their Freeform series Dead of Summer , Norman Buckley , Mairzee Almas , and Michael Schultz helm episodes of the season . Jennifer Lynch directed
the eighth episode of the season , `` I 'll Be Your Mirror '' .
</item 1>

<item 2>
Once Upon a Time (season 6)
In April 2017 , Horowitz and Kitsis confirmed that a group of characters will indeed have their stories wrapped up by the end of the season in acknowledging the
potential cast changes , saying : `` We planned this finale from the beginning of the year , so whoever stays and whoever goes ... all those questions have
already been dealt with . The audience does not have to fear ( anything feeling ) incomplete . '' On May 8 , Morrison confirmed that she had declined an offer
to remain on the show and that the sixth season would be her last , signaling the end of Emma 's time on the show as the main protagonist . On May 11 , Rebecca
Mader announced that she would also be leaving the show at the end of the season , citing creative decisions beyond her control . On May 12 , season six was
announced to be the last for four additional main actors : Ginnifer Goodwin , Josh Dallas , Jared Gilmore , and Emilie de Ravin . Thus , characters seeing their
storylines wrapped up in the season six finale include Emma Swan , Snow White , Prince Charming , Belle , and Zelena .
</item 2>

<item 3>
Once Upon a Time (season 6)
Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle
decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12
) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after
he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's
execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook
becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried
about whether David will give him his blessing to marry Emma , but David does .
</item 3>

<item 4>
Once Upon a Time (season 6)
Cast and characters ( edit ) Main article : List of Once Upon a Time characters Regular ( edit ) Ginnifer Goodwin as Snow White / Mary Margaret Blanchard
Jennifer Morrison as Emma Swan Lana Parrilla as Evil Queen / Regina Mills Josh Dallas as Prince Charming / David Nolan Emilie de Ravin as Belle / Lacey Colin
O'Donoghue as Captain Hook / Killian Jones Jared S. Gilmore as Henry Mills Rebecca Mader as Wicked Witch of the West / Zelena Robert Carlyle as Rumplestiltskin
/ Mr. Gold Recurring ( edit ) Beverley Elliott as Widow Lucas / Granny Raphael Sbarge as Jiminy Cricket / Dr. Archie Hopper Lee Arenberg as Dreamy / Grumpy /
Leroy David Paul Grove as Doc Gabe Khouth as Sneezy / Mr. Clark Faustino Di Bauda as Sleepy / Walter Mig Macario as Bashful Geoff Gustafson as Stealthy Tony
Amendola as Geppetto / Marco Keegan Connor Tracy as the Blue Fairy / Mother Superior Eion Bailey as Pinocchio / August Booth David Anders as Victor Frankenstein
/ Dr.
</item 4>

<item 5>
Once Upon a Time (season 6)
Existing fictional characters introduced to the series during the season include Aladdin , Princess Jasmine , the Count of Monte Cristo , Captain Nemo , Lady
Tremaine , Beowulf , Tiger Lily and the Tin Man . Original new characters include Gideon , the Black Fairy , Mary Lydgate , and Robert . The show also
reintroduced Jafar and Dr. Arthur Lydgate , who previously appeared in Once Upon a Time in Wonderland . This season also marks the final appearance of Emma Swan
( Jennifer Morrison ) as a series regular . Morrison announced she would be departing the series after the sixth - season finale , but if the series receives a
seventh season renewal she has agreed to appear in at least one episode . After serving as a series regular for two seasons , Rebecca Mader also announced that
season six would be her last on the show as a regular . Shortly after , Adam Horowitz and Edward Kitsis announced that original cast members Ginnifer Goodwin ,
Josh Dallas , and Jared Gilmore , as well as Emilie de Ravin who joined the main cast in season two , would also exit the show at the end of the season .
</item 5>

<item 6>
Once Upon a Time (season 6)
132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters
Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle
begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are
destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at
Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona
convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle .
</item 6>

<item 7>
Once Upon a Time (season 6)
Once Upon a Time ( season 6 ) - wikipedia Once Upon a Time ( season 6 ) Jump to : navigation , search Once Upon a Time ( season 6 ) DVD cover Starring Ginnifer
Goodwin Jennifer Morrison Lana Parrilla Josh Dallas Emilie de Ravin Colin O'Donoghue Jared S. Gilmore Rebecca Mader Robert Carlyle Country of origin United
States No. of episodes 22 Release Original network ABC Original release September 25 , 2016 ( 2016 - 09 - 25 ) -- May 14 , 2017 ( 2017 - 05 - 14 ) Season
chronology ← Previous Season 5 Next → Season 7 List of Once Upon a Time episodes The sixth season of the American ABC fantasy - drama Once Upon a Time was
ordered on March 3 , 2016 . It debuted on September 25 , 2016 , and concluded on May 14 , 2017 . In January 2017 , it was stated that the sixth season would end
the main storyline , and for a seventh season , the series would be softly rebooted with a new storyline .
</item 7>

<item 8>
Once Upon a Time (season 6)
Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in
Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents
him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma
having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their
`` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity .
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
</item 8>

<item 9>
Once Upon a Time (season 6)
Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Regular 2.2 Recurring 2.3 Guest 3 Episodes 4 Production 4.1 Development 4.1. 1 Musical episode 4.2 Casting
5 Ratings 6 References 7 External links Premise ( edit ) The residents of Storybrooke are faced with threats from the Evil Queen and Mr. Hyde , both with
different agendas following the arrival of refugees from the Land of Untold Stories . This leads to events to test Emma 's savior abilities that results in the
arrival of the Black Fairy to begin the Final Battle that was prophesied centuries ago . As dark and light collides , a new , and darker curse is unleashed ,
leading to the fall of all the realms with magic . The Final Battle is fought and won , leading to the heroes earning their happy beginnings , but a new
adventure begins in the far future for an adult Henry involving his daughter Lucy , and another curse looming across the New Enchanted Forest inhabitants .
</item 9>

<item 10>
Once Upon a Time (season 6)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz
September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In
Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of
a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him .
`` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied
about Robin Hood 's soul having been destroyed .
</item 10>
=== END ITEM id=a177-7a4a864abdfa1f7e ===

=== ITEM id=a164-c86b342167610132 ===
Gold sentences and facts:
Sentence 0: Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France `` for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy '' Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the properties of matter at low temperatures which led , inter alia , to the production of liquid helium '' 1914 Max von Laue Germany `` For his discovery of the diffraction of X-rays by crystals '' , an important step in the development of X-ray spectroscopy . 1915 William Henry Bragg United Kingdom `` For their services in the analysis of crystal structure by means of X-rays '' , an important step in the development of X-ray crystallography William Lawrence Bragg Australia United Kingdom 1916 Not awarded World War I 1917 Charles Glover Barkla United Kingdom `` For his discovery of the characteristic Röntgen radiation of the elements '' , another important step in the development of X-ray spectroscopy 1918 Max Planck Germany `` for the services he rendered to the advancement of physics by his discovery of energy quanta '' 1919 Johannes Stark Germany `` for his discovery of the Doppler effect in canal rays and the splitting of spectral lines in electric fields '' 1920 Charles Édouard Guillaume Switzerland `` for the service he has rendered to precision measurements in physics by his discovery of anomalies in nickel - steel alloys '' 1921 Albert Einstein Germany Switzerland `` for his services to theoretical physics , and especially for his discovery of the law of the photoelectric effect '' 1922 Niels Bohr Denmark `` for his services in the investigation of the structure of atoms and of the radiation emanating from them '' 1923 Robert Andrews Millikan United States `` for his work on the elementary charge of electricity and on the photoelectric effect '' 1924 Manne Siegbahn Sweden `` for his discoveries and research in the field of X-ray spectroscopy '' 1925 James Franck Germany `` for their discovery of the laws governing the impact of an electron upon an atom '' Gustav Hertz Germany 1926 Jean Baptiste Perrin France `` for his work on the discontinuous structure of matter , and especially for his discovery of sedimentation equilibrium '' 1927 Arthur Holly Compton United States `` for his discovery of the effect named after him '' Charles Thomson Rees Wilson United Kingdom `` for his method of making the paths of electrically charged particles visible by condensation of vapour '' 1928 Owen Willans Richardson United Kingdom `` for his work on the thermionic phenomenon and especially for the discovery of the law named after him '' 1929 Louis Victor Pierre Raymond , 7th Duc de Broglie France `` for his discovery of the wave nature of electrons '' 1930 Chandrasekhara Venkata Raman India `` for his work on the scattering of light and for the discovery of the effect named after him '' 1931 Not awarded 1932 Werner Heisenberg Germany `` for the creation of quantum mechanics , the application of which has , inter alia , led to the discovery of the allotropic forms of hydrogen '' 1933 Erwin Schrödinger Austria `` for the discovery of new productive forms of atomic theory '' Paul Dirac United Kingdom 1934 Not awarded 1935 James Chadwick United Kingdom `` for the discovery of the neutron '' 1936 Victor Francis Hess Austria `` for his discovery of cosmic radiation '' Carl David Anderson United States `` for his discovery of the positron '' 1937 Clinton Joseph Davisson United States `` for their experimental discovery of the diffraction of electrons by crystals '' George Paget Thomson United Kingdom 1938 Enrico Fermi Italy `` for his demonstrations of the existence of new radioactive elements produced by neutron irradiation , and for his related discovery of nuclear reactions brought about by slow neutrons '' 1939 Ernest Lawrence United States `` for the invention and development of the cyclotron and for results obtained with it , especially with regard to artificial radioactive elements '' 1940 Not awarded World War II 1941 Not awarded World War II 1942 Not awarded World War II 1943 Otto Stern United States `` for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton '' 1944 Isidor Isaac Rabi United States Poland `` for his resonance method for recording the magnetic properties of atomic nuclei '' 1945 Wolfgang Pauli Austria `` for the discovery of the Exclusion Principle , also called the Pauli principle '' 1946 Percy Williams Bridgman United States `` for the invention of an apparatus to produce extremely high pressures , and for the discoveries he made there within the field of high pressure physics '' 1947 Edward Victor Appleton United Kingdom `` for his investigations of the physics of the upper atmosphere especially for the discovery of the so - called Appleton layer '' 1948 Patrick Maynard Stuart Blackett United Kingdom `` for his development of the Wilson cloud chamber method , and his discoveries therewith in the fields of nuclear physics and cosmic radiation '' 1949 Hideki Yukawa Japan `` for his prediction of the existence of mesons on the basis of theoretical work on nuclear forces '' 1950 Cecil Frank Powell United Kingdom `` for his development of the photographic method of studying nuclear processes and his discoveries regarding mesons made with this method '' 1951 John Douglas Cockcroft United Kingdom `` for their pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles '' Ernest Thomas Sinton Walton Ireland 1952 Felix Bloch Switzerland United States `` for their development of new methods for nuclear magnetic precision measurements and discoveries in connection therewith '' Edward Mills Purcell United States 1953 Frits Zernike Netherlands `` for his demonstration of the phase contrast method , especially for his invention of the phase contrast microscope '' 1954 Max Born West Germany `` for his fundamental research in quantum mechanics , especially for his statistical interpretation of the wavefunction '' Walther Bothe West Germany `` for the coincidence method and his discoveries made therewith '' 1955 Willis Eugene Lamb United States `` for his discoveries concerning the fine structure of the hydrogen spectrum '' Polykarp Kusch United States Germany `` for his precision determination of the magnetic moment of the electron '' 1956 John Bardeen United States `` for their researches on semiconductors and their discovery of the transistor effect '' Walter Houser Brattain United States William Bradford Shockley United States 1957 Tsung - Dao Lee Republic of China `` for their penetrating investigation of the so - called parity laws which has led to important discoveries regarding the elementary particles '' Chen Ning Yang Republic of China 1958 Pavel Alekseyevich Cherenkov Soviet Union `` for the discovery and the interpretation of the Cherenkov effect '' Ilya Frank Soviet Union Igor Yevgenyevich Tamm Soviet Union 1959 Emilio Gino Segrè Italy United States `` for their discovery of the antiproton '' Owen Chamberlain United States 1960 Donald Arthur Glaser United States `` for the invention of the bubble chamber '' 1961 Robert Hofstadter United States `` for his pioneering studies of electron scattering in atomic nuclei and for his thereby achieved discoveries concerning the structure of the nucleons '' Rudolf Ludwig Mössbauer West Germany `` for his researches concerning the resonance absorption of gamma radiation and his discovery in this connection of the effect which bears his name '' 1962 Lev Davidovich Landau Soviet Union `` for his pioneering theories for condensed matter , especially liquid helium '' 1963 Eugene Paul Wigner Hungary United States `` for his contributions to the theory of the atomic nucleus and the elementary particles , particularly through the discovery and application of fundamental symmetry principles '' Maria Goeppert - Mayer United States `` for their discoveries concerning nuclear shell structure '' J. Hans D. Jensen West Germany 1964 Nicolay Gennadiyevich Basov Soviet Union `` for fundamental work in the field of quantum electronics , which has led to the construction of oscillators and amplifiers based on the maser -- laser principle '' Alexander Prokhorov Soviet Union Charles Hard Townes United States 1965 Richard Phillips Feynman United States `` for their fundamental work in quantum electrodynamics ( QED ) , with deep - ploughing consequences for the physics of elementary particles '' Julian Schwinger United States Shin'ichirō Tomonaga Japan 1966 Alfred Kastler France `` for the discovery and development of optical methods for studying Hertzian resonances in atoms '' 1967 Hans Albrecht Bethe United States `` for his contributions to the theory of nuclear reactions , especially his discoveries concerning the energy production in stars '' 1968 Luis Walter Alvarez United States `` for his decisive contributions to elementary particle physics , in particular the discovery of a large number of resonance states , made possible through his development of the technique of using hydrogen bubble chamber and data analysis '' 1969 Murray Gell - Mann United States `` for his contributions and discoveries concerning the classification of elementary particles and their interactions '' 1970 Hannes Olof Gösta Alfvén Sweden `` for fundamental work and discoveries in magneto - hydrodynamics with fruitful applications in different parts of plasma physics '' Louis Néel France `` for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism which have led to important applications in solid state physics '' 1971 Dennis Gabor Hungary United Kingdom `` for his invention and development of the holographic method '' 1972 John Bardeen United States `` for their jointly developed theory of superconductivity , usually called the BCS - theory '' Leon Neil Cooper United States John Robert Schrieffer United States 1973 Leo Esaki Japan `` for their experimental discoveries regarding tunneling phenomena in semiconductors and superconductors , respectively '' Ivar Giaever United States Norway Brian David Josephson United Kingdom `` for his theoretical predictions of the properties of a supercurrent through a tunnel barrier , in particular those phenomena which are generally known as the Josephson effect '' Martin Ryle United Kingdom `` for their pioneering research in radio astrophysics : Ryle for his observations and inventions , in particular of the aperture synthesis technique , and Hewish for his decisive role in the discovery of pulsars '' Antony Hewish United Kingdom Aage Bohr Denmark `` for the discovery of the connection between collective motion and particle motion in atomic nuclei and the development of the theory of the structure of the atomic nucleus based on this connection '' Ben Roy Mottelson Denmark Leo James Rainwater United States 1976 Burton Richter United States `` for their pioneering work in the discovery of a heavy elementary particle of a new kind '' Samuel Chao Chung Ting United States 1977 Philip Warren Anderson United States `` for their fundamental theoretical investigations of the electronic structure of magnetic and disordered systems '' Nevill Francis Mott United Kingdom John Hasbrouck Van Vleck United States 1978 Pyotr Leonidovich Kapitsa Soviet Union `` for his basic inventions and discoveries in the area of low - temperature physics '' Arno Allan Penzias United States `` for their discovery of cosmic microwave background radiation '' Robert Woodrow Wilson United States 1979 Sheldon Lee Glashow United States `` for their contributions to the theory of the unified weak and electromagnetic interaction between elementary particles , including , inter alia , the prediction of the weak neutral current '' Abdus Salam Pakistan Steven Weinberg United States 1980 James Watson Cronin United States `` for the discovery of violations of fundamental symmetry principles in the decay of neutral K - mesons '' Val Logsdon Fitch United States 1981 Nicolaas Bloembergen Netherlands United States `` for their contribution to the development of laser spectroscopy '' Arthur Leonard Schawlow United States Kai Manne Börje Siegbahn Sweden `` for his contribution to the development of high - resolution electron spectroscopy '' 1982 Kenneth G. Wilson United States `` for his theory for critical phenomena in connection with phase transitions '' Subrahmanyan Chandrasekhar India United States `` for his theoretical studies of the physical processes of importance to the structure and evolution of the stars '' William Alfred Fowler United States `` for his theoretical and experimental studies of the nuclear reactions of importance in the formation of the chemical elements in the universe '' 1984 Carlo Rubbia Italy `` for their decisive contributions to the large project , which led to the discovery of the field particles W and Z , communicators of weak interaction '' Simon van der Meer Netherlands 1985 Klaus von Klitzing West Germany `` for the discovery of the quantized Hall effect '' 1986 Ernst Ruska West Germany `` for his fundamental work in electron optics , and for the design of the first electron microscope '' Gerd Binnig West Germany `` for their design of the scanning tunneling microscope '' Heinrich Rohrer Switzerland Johannes Georg Bednorz West Germany `` for their important break - through in the discovery of superconductivity in ceramic materials '' Karl Alexander Müller Switzerland 1988 Leon Max Lederman United States `` for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino '' Melvin Schwartz United States Jack Steinberger United States 1989 Norman Foster Ramsey United States `` for the invention of the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks '' Hans Georg Dehmelt United States Germany `` for the development of the ion trap technique '' Wolfgang Paul West Germany 1990 Jerome I. Friedman United States `` for their pioneering investigations concerning deep inelastic scattering of electrons on protons and bound neutrons , which have been of essential importance for the development of the quark model in particle physics '' Henry Way Kendall United States Richard E. Taylor Canada 1991 Pierre - Gilles de Gennes France `` for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter , in particular to liquid crystals and polymers '' 1992 Georges Charpak France Poland `` for his invention and development of particle detectors , in particular the multiwire proportional chamber '' 1993 Russell Alan Hulse United States `` for the discovery of a new type of pulsar , a discovery that has opened up new possibilities for the study of gravitation '' Joseph Hooton Taylor , Jr . United States 1994 Bertram Brockhouse Canada `` for the development of neutron spectroscopy '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' Clifford Glenwood Shull United States `` for the development of the neutron diffraction technique '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' 1995 Martin Lewis Perl United States `` for the discovery of the tau lepton '' and `` for pioneering experimental contributions to lepton physics '' Frederick Reines United States `` for the detection of the neutrino '' and `` for pioneering experimental contributions to lepton physics '' David Morris Lee United States `` for their discovery of superfluidity in helium - 3 '' Douglas D. Osheroff United States Robert Coleman Richardson United States 1997 Steven Chu United States `` for development of methods to cool and trap atoms with laser light . '' Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands `` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002 Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics , in particular for the detection of cosmic neutrinos '' Masatoshi Koshiba Japan Riccardo Giacconi Italy United States `` for pioneering contributions to astrophysics , which have led to the discovery of cosmic X-ray sources '' 2003 Alexei Alexeyevich Abrikosov Russia United States `` for pioneering contributions to the theory of superconductors and superfluids '' Vitaly Lazarevich Ginzburg Russia Anthony James Leggett United Kingdom United States David J. Gross United States `` for the discovery of asymptotic freedom in the theory of the strong interaction '' Hugh David Politzer United States Frank Wilczek United States 2005 Roy J. Glauber United States `` for his contribution to the quantum theory of optical coherence '' John L. Hall United States `` for their contributions to the development of laser - based precision spectroscopy , including the optical frequency comb technique '' Theodor W. Hänsch Germany 2006 John C. Mather United States `` for their discovery of the blackbody form and anisotropy of the cosmic microwave background radiation '' George F. Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and manipulation of individual quantum systems . '' David J. Wineland United States 2013 François Englert Belgium `` for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles , and which recently was confirmed through the discovery of the predicted fundamental particle , by the ATLAS and CMS experiments at CERN 's Large Hadron Collider '' Peter Higgs United Kingdom 2014 Isamu Akasaki Japan `` for the invention of efficient blue light - emitting diodes which has enabled bright and energy - saving white light sources '' Hiroshi Amano Japan Shuji Nakamura Japan United States 2015 Takaaki Kajita Japan `` for the discovery of neutrino oscillations , which shows that neutrinos have mass '' Arthur B. McDonald Canada 2016 David J. Thouless United Kingdom United States `` for theoretical discoveries of topological phase transitions and topological phases of matter '' F. Duncan M. Haldane United Kingdom United States John M. Kosterlitz United Kingdom United States 2017 Rainer Weiss Germany United States `` for decisive contributions to the LIGO detector and the observation of gravitational waves '' Kip Thorne United States Barry Barish United States
  Fact 0.46: 1936: Victor Francis Hess of Austria won the physics prize for discovering cosmic radiation

Retrieved items:
<item 1>
List of Nobel laureates in Physics
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1971 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1972 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1973 '' . Nobel Foundation .
Archived from the original on 2011 - 06 - 27 . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1974 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1975 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1976 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 1>

<item 2>
List of Nobel laureates in Physics
As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934
, 1940 -- 1942 ) . Laureates ( edit ) Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has
rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service
they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France ``
for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri
Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and
for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays ''
1906 Joseph
</item 2>

<item 3>
The list surveys Physics Nobel laureates from 1906 to 2005, often sharing awards. Honored breakthroughs span cathode rays, electricity, X-rays, quantum theory,
atomic and nuclear physics, relativity, particle physics, superconductivity, and cosmology. Early citations highlight Thomson, Einstein, Bohr, Planck, and
others; later rosters include Feynman, Yang, Gell-Mann, Wilson, and many more. No prizes were awarded in 191
</item 3>

<item 4>
The Nobel Prize in Physics, established under Alfred Nobel’s 1895 will, is awarded annually by the Royal Swedish Academy of Sciences and presented in Stockholm
on December 10. First awarded to Wilhelm Röntgen in 1901, it recognizes outstanding contributions to physics. By 2017, 206 people had received it; no prize was
awarded in six years. John Bardeen is the only two-time physics laureate. Marie Curie and Maria Goeppert-Mayer are the
</item 4>

<item 5>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1925 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1926 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics 1927 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1928 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1929 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1930 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1932 ''
. Nobel Foundation . Retrieved 2008 - 10 - 09 .
</item 5>

<item 6>
List of Nobel laureates in Physics
List of Nobel laureates in Physics - wikipedia List of Nobel laureates in Physics Jump to : navigation , search Front side ( obverse ) of the Nobel Prize Medal
for Physics presented to Edward Victor Appleton in 1947 The Nobel Prize in Physics ( Swedish : Nobelpriset i fysik ) is awarded annually by the Royal Swedish
Academy of Sciences to scientists in the various fields of physics . It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel ( who died
in 1896 ) , awarded for outstanding contributions in physics . As dictated by Nobel 's will , the award is administered by the Nobel Foundation and awarded by a
committee that consists of five members elected by the Royal Swedish Academy of Sciences . The award is presented in Stockholm at an annual ceremony on December
10 , the anniversary of Nobel 's death . Each recipient receives a medal , a diploma and a monetary award prize that has varied throughout the years .
</item 6>

<item 7>
List of Nobel laureates in Physics
Jump up ^ `` The Nobel Prize in Physics 1918 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1919 '' . Nobel
Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1920 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel
Prize in Physics 1921 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1922 '' . Nobel Foundation . Retrieved 2008 -
10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1923 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1924 '' .
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1925 '' .
</item 7>

<item 8>
List of Nobel laureates in Physics
This information may not necessarily reflect the recipient 's birthplace or citizenship . ^ C. The citation for each award is quoted ( not always in full ) from
nobelprize.org , the official website of the Nobel Foundation . The links in this column are to articles ( or sections of articles ) on the history and areas of
physics for which the awards were presented . The links are intended only as a guide and explanation . For a full account of the work done by each Nobel
laureate , please see the biography articles linked from the name column . Citations ( edit ) Jump up ^ `` Alfred Nobel -- The Man Behind the Nobel Prize '' .
Nobel Foundation . Retrieved 2008 - 10 - 29 . Jump up ^ `` The Nobel Prize Awarders '' . Nobel Foundation . Archived from the original on 2008 - 10 - 15 .
Retrieved 2008 - 10 - 29 . Jump up ^ `` The Nobel Prize Award Ceremonies '' . Nobel Foundation . Archived from the original on 2008 - 08 - 22 .
</item 8>

<item 9>
List of Nobel laureates in Physics
Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1957 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The
Nobel Prize in Physics 1958 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1959 '' . Nobel Foundation . Retrieved
2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1960 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . ^ Jump up to : `` The Nobel Prize in Physics
1961 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . Jump up ^ `` The Nobel Prize in Physics 1962 '' . Nobel Foundation . Retrieved 2008 - 10 - 09 . ^ Jump
up to : `` The Nobel Prize in Physics 1963 '' . Nobel Foundation .
</item 9>

<item 10>
List of Nobel laureates in Physics
Contents ( hide ) 1 Statistics 2 Laureates 3 See also 4 References 4.1 Notes 4.2 Citations 4.3 Sources 5 External links Statistics ( edit ) The first Nobel
Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John
Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry
in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won
the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals .
</item 10>
=== END ITEM id=a164-c86b342167610132 ===

=== ITEM id=a152-6f0d15b5824c2475 ===
Gold sentences and facts:
Sentence 0: Season Team League Conference Division Regular season Postseason Results Awards Head coaches Finish Wins Losses Ties 1960 1960 NFL Western 7th 0 11 Tom Landry 1961 1961 NFL Eastern 6th 9 Tom Landry 1962 1962 NFL Eastern 5th 5 8 Tom Landry 1963 1963 NFL Eastern 5th 10 0 Tom Landry 1964 1964 NFL Eastern 5th 5 8 Tom Landry 1965 1965 NFL Eastern 2nd 7 7 0 Tom Landry 1966 1966 NFL Eastern 1st 10 Lost NFL Championship Game ( Packers ) 34 -- 27 Tom Landry ( COY ) Tom Landry 1967 1967 NFL Eastern Capitol 1st 9 5 0 Won Conference Playoffs ( Browns ) 52 -- 14 Lost NFL Championship Game ( Packers ) 21 -- 17 Tom Landry 1968 1968 NFL Eastern Capitol 1st 12 0 Lost Conference Playoffs ( Browns ) 31 -- 20 Tom Landry 1969 1969 NFL Eastern Capitol 1st 11 Lost Conference Playoffs ( Browns ) 38 -- 14 Calvin Hill ( OROY ) Tom Landry 1970 1970 NFL NFC East 1st 10 0 Won Divisional Playoffs ( Lions ) 5 -- 0 Won Conference Championship ( 49ers ) 17 -- 10 Lost Super Bowl V ( Colts ) 16 -- 13 Chuck Howley ( SB MVP ) Tom Landry 1971 1971 NFL NFC East 1st 11 0 Won Divisional Playoffs ( Vikings ) 20 -- 12 Won Conference Championship ( 49ers ) 14 -- 3 Won Super Bowl VI ( 1 ) ( Dolphins ) 24 -- 3 Roger Staubach ( SB MVP ) Tom Landry 1972 1972 NFL NFC East 2nd 10 0 Won Divisional Playoffs ( 49ers ) 30 -- 28 Lost Conference Championship ( Redskins ) 26 -- 3 Tom Landry 1973 1973 NFL NFC East 1st 10 0 Won Divisional Playoffs ( Rams ) 27 -- 26 Lost Conference Championship ( Vikings ) 27 -- 10 Tom Landry NFL NFC East 3rd 8 6 0 Tom Landry NFL NFC East 2nd 10 0 Won Divisional Playoffs ( Vikings ) 17 -- 14 Won Conference Championship ( Rams ) 37 -- 7 Lost Super Bowl X ( Steelers ) 21 -- 17 Tom Landry 1976 1976 NFL NFC East 1st 11 0 Lost Divisional Playoffs ( Rams ) 14 -- 12 Tom Landry 1977 1977 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Bears ) 37 -- 7 Won Conference Championship ( Vikings ) 23 -- 6 Won Super Bowl XII ( 2 ) ( Broncos ) 27 -- 10 Tony Dorsett ( OROY ) Harvey Martin ( DPOY , SB MVP ) Randy White ( SB MVP ) Tom Landry 1978 1978 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Falcons ) 27 -- 20 Won Conference Championship ( Rams ) 28 -- 0 Lost Super Bowl XIII ( Steelers ) 35 -- 31 Tom Landry 1979 1979 NFL NFC East 1st 11 5 0 Lost Divisional Playoffs ( Rams ) 21 -- 19 Tom Landry 1980 1980 NFL NFC East 2nd 12 0 Won Wild Card Playoffs ( Rams ) 34 -- 17 Won Divisional Playoffs ( Falcons ) 30 -- 27 Lost Conference Championship ( Eagles ) 20 -- 7 Tom Landry 1981 1981 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Buccaneers ) 38 -- 0 Lost Conference Championship ( 49ers ) 28 -- 27 Tom Landry 1982 1982 NFL NFC 2nd 6 0 Won Wild Card Playoffs ( Buccaneers ) 30 -- 17 Won Divisional Playoffs ( Packers ) 37 -- 26 Lost Conference Championship ( Redskins ) 31 -- 17 Tom Landry NFL NFC East 2nd 12 0 Lost Wild Card Playoffs ( Rams ) 24 -- 17 Tom Landry 1984 1984 NFL NFC East 4th 9 7 0 Tom Landry 1985 1985 NFL NFC East 1st 10 6 0 Lost Divisional Playoffs ( Rams ) 20 -- 0 Tom Landry 1986 1986 NFL NFC East 3rd 7 9 0 Tom Landry NFL NFC East 4th 7 8 0 Tom Landry 1988 1988 NFL NFC East 5th 13 0 Tom Landry 1989 1989 NFL NFC East 5th 15 0 Jimmy Johnson 1990 1990 NFL NFC East 4th 7 9 0 Emmitt Smith ( OROY ) Jimmy Johnson ( COY ) Jimmy Johnson 1991 1991 NFL NFC East 2nd 11 5 0 Won Wild Card Playoffs ( Bears ) 17 -- 13 Lost Divisional Playoffs ( Lions ) 38 -- 6 Jimmy Johnson 1992 1992 NFL NFC East 1st 13 0 Won Divisional Playoffs ( Eagles ) 34 -- 10 Won Conference Championship ( 49ers ) 30 -- 20 Won Super Bowl XXVII ( 3 ) ( Bills ) 52 -- 17 Troy Aikman ( SB MVP ) Jimmy Johnson 1993 1993 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Packers ) 27 -- 17 Won Conference Championship ( 49ers ) 38 -- 21 Won Super Bowl XXVIII ( 4 ) ( Bills ) 30 -- 13 Emmitt Smith ( MVP , SB MVP ) Jimmy Johnson 1994 1994 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Packers ) 35 -- 9 Lost Conference Championship ( 49ers ) 38 -- 28 Barry Switzer 1995 1995 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Eagles ) 30 -- 11 Won Conference Championship ( Packers ) 38 -- 27 Won Super Bowl XXX ( 5 ) ( Steelers ) 27 -- 17 Larry Brown ( SB MVP ) Barry Switzer NFL NFC East 1st 10 6 0 Won Wild Card Playoffs ( Vikings ) 40 -- 15 Lost Divisional Playoffs ( Panthers ) 26 -- 17 Barry Switzer 1997 1997 NFL NFC East 4th 6 10 0 Barry Switzer 1998 1998 NFL NFC East 1st 10 6 0 Lost Wild Card Playoffs ( Cardinals ) 20 -- 7 Chan Gailey 1999 1999 NFL NFC East 2nd 8 8 0 Lost Wild Card Playoffs ( Vikings ) 27 -- 10 Chan Gailey 2000 2000 NFL NFC East 4th 5 11 0 Dave Campo 2001 2001 NFL NFC East 5th 5 11 0 Dave Campo 2002 2002 NFL NFC East 4th 5 11 0 Dave Campo 2003 2003 NFL NFC East 2nd 10 6 0 Lost Wild Card Playoffs ( Panthers ) 29 -- 10 Bill Parcells NFL NFC East 3rd 6 10 0 Bill Parcells 2005 2005 NFL NFC East 3rd 9 7 0 Bill Parcells 2006 2006 NFL NFC East 2nd 9 7 0 Lost Wild Card Playoffs ( Seahawks ) 21 -- 20 Bill Parcells 2007 2007 NFL NFC East 1st 13 0 Lost Divisional Playoffs ( Giants ) 21 -- 17 Greg Ellis ( CBPOY ) Wade Phillips 2008 2008 NFL NFC East 3rd 9 7 0 Wade Phillips 2009 2009 NFL NFC East 1st 11 5 0 Won Wild Card Playoffs ( Eagles ) 34 -- 14 Lost Divisional Playoffs ( Vikings ) 34 -- 3 Wade Phillips NFL NFC East 3rd 6 10 0 Wade Phillips ( 1 -- 7 ) Jason Garrett ( 5 -- 3 ) 2011 2011 NFL NFC East 3rd 8 8 0 Jason Garrett 2012 2012 NFL NFC East 3rd 8 8 0 Jason Witten ( WP MOY ) Jason Garrett 2013 2013 NFL NFC East 2nd 8 8 0 Jason Garrett 2014 2014 NFL NFC East 1st 12 0 Won Wild Card Playoffs ( Lions ) 24 -- 20 Lost Divisional Playoffs ( Packers ) 26 -- 21 DeMarco Murray ( OPOY ) Jason Garrett 2015 2015 NFL NFC East 4th 12 0 Jason Garrett 2016 2016 NFL NFC East 1st 13 0 Lost Divisional Playoffs ( Packers ) 34 -- 31 Dak Prescott ( OROY ) Jason Garrett ( COY ) Jason Garrett 2017 2017 NFL NFC East 2nd 9 7 0 Jason Garrett Total 502 374 6 All - time regular season record ( 1960 -- 2017 ) 34 27 0 All - time postseason record ( 1960 -- 2017 ) 536 401 6 All - time regular season and postseason record ( 1960 -- 2017 ) 5 NFL Championships , 10 Conference Championships , 22 Divisional Championships
  Fact 0.57: In 1980, the team lost the Conference Championship to the Eagles, 20-7.

Retrieved items:
<item 1>
List of Dallas Cowboys seasons
Conference Playoffs ( Browns ) 38 -- 14 Calvin Hill ( OROY ) Tom Landry 1970 1970 NFL NFC East 1st 10 0 Won Divisional Playoffs ( Lions ) 5 -- 0 Won Conference
Championship ( 49ers ) 17 -- 10 Lost Super Bowl V ( Colts ) 16 -- 13 Chuck Howley ( SB MVP ) Tom Landry 1971 1971 NFL NFC East 1st 11 0 Won Divisional Playoffs
( Vikings ) 20 -- 12 Won Conference Championship ( 49ers ) 14 -- 3 Won Super Bowl VI ( 1 ) ( Dolphins ) 24 -- 3 Roger Staubach ( SB MVP ) Tom Landry 1972 1972
NFL NFC East 2nd 10 0 Won Divisional Playoffs ( 49ers ) 30 -- 28 Lost Conference Championship ( Redskins ) 26 -- 3 Tom Landry 1973 1973 NFL NFC East 1st 10 0
Won Divisional Playoffs ( Rams ) 27 -- 26 Lost Conference Championship ( Vikings ) 27 -- 10 Tom Landry NFL NFC East 3rd 8
</item 1>

<item 2>
List of Dallas Cowboys seasons
They also played in and lost Super Bowl V , X , and XIII . The franchise has experienced two major periods of continued success in their history . The first
period of success came from 1966 -- 1985 when the Cowboys played in the postseason 18 times . During this period , they played in two NFL Championships and five
Super Bowls , winning two of them , winning a total of 20 playoff games . The second period of success was between 1991 -- 1996 when the Cowboys captured five
straight NFC East Division titles and won three Super Bowls going 11 -- 2 in the postseason . The Cowboys have also experienced failure in their history . The
most notable period of failure was from their 1960 inaugural season to 1965 , during which the Cowboys did not have a single postseason appearance . They did
not win a single game during their first season , compiling an 0 -- 11 -- 1 record that is still the worst in franchise history . Also , they did not have a
single winning record in this period .
</item 2>

<item 3>
Founded in 1960, the Dallas Cowboys’ season history through 2017 includes five Super Bowl victories (VI, XII, XXVII, XXVIII and XXX) and three Super Bowl
losses. They made 31 postseason appearances, reached 14 NFC Championship Games and recorded 20 consecutive winning seasons. Their strongest eras were 1966–85
and 1991–96, including three Super Bowl wins in the latter period. Early struggles included an 0–11–1
</item 3>

<item 4>
List of Dallas Cowboys seasons
Playoffs ( Eagles ) 34 -- 10 Won Conference Championship ( 49ers ) 30 -- 20 Won Super Bowl XXVII ( 3 ) ( Bills ) 52 -- 17 Troy Aikman ( SB MVP ) Jimmy Johnson
1993 1993 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Packers ) 27 -- 17 Won Conference Championship ( 49ers ) 38 -- 21 Won Super Bowl XXVIII ( 4 ) ( Bills
) 30 -- 13 Emmitt Smith ( MVP , SB MVP ) Jimmy Johnson 1994 1994 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Packers ) 35 -- 9 Lost Conference Championship
( 49ers ) 38 -- 28 Barry Switzer 1995 1995 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Eagles ) 30 -- 11 Won Conference Championship ( Packers ) 38 -- 27
Won Super Bowl XXX ( 5 ) ( Steelers ) 27 -- 17 Larry Brown ( SB MVP ) Barry Switzer NFL NFC East 1st 10 6 0 Won Wild Card Playoffs
</item 4>

<item 5>
List of Dallas Cowboys seasons
East 3rd 6 10 0 Bill Parcells 2005 2005 NFL NFC East 3rd 9 7 0 Bill Parcells 2006 2006 NFL NFC East 2nd 9 7 0 Lost Wild Card Playoffs ( Seahawks ) 21 -- 20 Bill
Parcells 2007 2007 NFL NFC East 1st 13 0 Lost Divisional Playoffs ( Giants ) 21 -- 17 Greg Ellis ( CBPOY ) Wade Phillips 2008 2008 NFL NFC East 3rd 9 7 0 Wade
Phillips 2009 2009 NFL NFC East 1st 11 5 0 Won Wild Card Playoffs ( Eagles ) 34 -- 14 Lost Divisional Playoffs ( Vikings ) 34 -- 3 Wade Phillips NFL NFC East
3rd 6 10 0 Wade Phillips ( 1 -- 7 ) Jason Garrett ( 5 -- 3 ) 2011 2011 NFL NFC East 3rd 8 8 0 Jason Garrett 2012 2012 NFL NFC East 3rd 8 8 0 Jason Witten ( WP
MOY ) Jason
</item 5>

<item 6>
List of Dallas Cowboys seasons
Also , they did not have a single winning record in this period . Between 1986 and 1990 the Cowboys had losing records in each season as veteran coach Landry
retired and the team was radically overhauled , with the low point being the NFL 's second 15 - loss season ( after the 1980 Saints ) in 1989 . After losing a
Divisional playoff Game in 1996 , the Cowboys between 1997 and 2008 lost five consecutive playoff games , one after a franchise - record 13 -- 3 season in 2007
, during which most predicted the Cowboys would break this streak . This streak finally came to an end when the Cowboys finally beat their bitter rival , the
Philadelphia Eagles 34 -- 14 after an 11 -- 5 season in 2009 . Seasons ( edit ) NFL Champions ( 1920 -- 1969 ) Super Bowl Champions ( 1970 -- present )
Conference Champions Division Champions Wild Card Berth Season Team League Conference Division Regular season Postseason Results Awards Head coaches Finish Wins
Losses Ties 1960 1960 NFL Western 7th 0 11 Tom Landry 1961 1961 NFL Eastern
</item 6>

<item 7>
List of Dallas Cowboys seasons
1992 1993 1994 1995 1998 2007 2009 2014 2016 Conference championships ( 10 ) 1966 1967 1970 1971 1977 1978 1992 1993 1995 League Championships ( 5 ) 1971 ( VI )
1977 ( XII ) 1992 ( XXVII ) 1993 ( XXVIII ) 1995 ( XXX ) Media Broadcasters Radio network KRLD - FM Brad Sham Babe Laufenberg Current league affiliations League
: National Football League Conference : National Football Conference Division : East Division Seasons ( 58 ) 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969
1970 1971 1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000
</item 7>

<item 8>
List of Dallas Cowboys seasons
Pro football reference.com . Retrieved April 14 , 2012 . `` Dallas Cowboys Live -- Dallas Cowboys '' . cowboysgames.net . Retrieved July 1 , 2016 . Dallas
Cowboys Founded in 1960 Based in Arlington , Texas Headquartered in Frisco , Texas Franchise Franchise History Coaches Players Seasons Expansion draft Draft
history Starting quarterbacks First - round draft picks The Ford Center at The Star Stadiums Cotton Bowl Texas Stadium AT&T Stadium Culture NFL on Thanksgiving
Day America 's Team Cheerleaders Doomsday Defense Crazy Ray Rowdy Jerry Jones Black Sunday ( film ) King of the Hill Lore Tom Landry Tex Schramm Captain
Comeback Ice Bowl Hail Mary The Catch Herschel Walker trade Dirty Dozen Ring of Honor Bounty Bowl series Rivalries New York Giants Philadelphia Eagles
Washington Redskins Green Bay Packers San Francisco 49ers Pittsburgh Steelers Houston Oilers / Texans Division championships ( 23 ) 1967 1968 1969 1970 1971
1972 1973 1976 1977 1978 1979 1981 1985 1992 1993 1994 1995 1998 2007
</item 8>

<item 9>
List of Dallas Cowboys seasons
NFC East 3rd 8 8 0 Jason Witten ( WP MOY ) Jason Garrett 2013 2013 NFL NFC East 2nd 8 8 0 Jason Garrett 2014 2014 NFL NFC East 1st 12 0 Won Wild Card Playoffs (
Lions ) 24 -- 20 Lost Divisional Playoffs ( Packers ) 26 -- 21 DeMarco Murray ( OPOY ) Jason Garrett 2015 2015 NFL NFC East 4th 12 0 Jason Garrett 2016 2016 NFL
NFC East 1st 13 0 Lost Divisional Playoffs ( Packers ) 34 -- 31 Dak Prescott ( OROY ) Jason Garrett ( COY ) Jason Garrett 2017 2017 NFL NFC East 2nd 9 7 0 Jason
Garrett Total 502 374 6 All - time regular season record ( 1960 -- 2017 ) 34 27 0 All - time postseason record ( 1960 -- 2017 ) 536 401 6 All - time regular
season and postseason record ( 1960 -- 2017 ) 5 NFL Championships , 10 Conference Championships ,
</item 9>

<item 10>
List of Dallas Cowboys seasons
0 Lost Super Bowl XIII ( Steelers ) 35 -- 31 Tom Landry 1979 1979 NFL NFC East 1st 11 5 0 Lost Divisional Playoffs ( Rams ) 21 -- 19 Tom Landry 1980 1980 NFL
NFC East 2nd 12 0 Won Wild Card Playoffs ( Rams ) 34 -- 17 Won Divisional Playoffs ( Falcons ) 30 -- 27 Lost Conference Championship ( Eagles ) 20 -- 7 Tom
Landry 1981 1981 NFL NFC East 1st 12 0 Won Divisional Playoffs ( Buccaneers ) 38 -- 0 Lost Conference Championship ( 49ers ) 28 -- 27 Tom Landry 1982 1982 NFL
NFC 2nd 6 0 Won Wild Card Playoffs ( Buccaneers ) 30 -- 17 Won Divisional Playoffs ( Packers ) 37 -- 26 Lost Conference Championship ( Redskins ) 31 -- 17 Tom
Landry NFL NFC East 2nd 12 0 Lost Wild Card Playoffs ( Rams ) 24 -- 17 Tom Landry 1984 1984 NFL NFC East 4th 9 7
</item 10>
=== END ITEM id=a152-6f0d15b5824c2475 ===

=== ITEM id=a192-0196e65757df2a04 ===
Gold sentences and facts:
Sentence 0: The earliest predecessors of the eye were photoreceptor proteins that sense light , found even in unicellular organisms , called `` eyespots '' . Eyespots can only sense ambient brightness : they can distinguish light from dark , sufficient for photoperiodism and daily synchronization of circadian rhythms . They are insufficient for vision , as they can not distinguish shapes or determine the direction light is coming from . Eyespots are found in nearly all major animal groups , and are common among unicellular organisms , including euglena . The euglena 's eyespot , called a stigma , is located at its anterior end . It is a small splotch of red pigment which shades a collection of light sensitive crystals . Together with the leading flagellum , the eyespot allows the organism to move in response to light , often toward the light to assist in photosynthesis , and to predict day and night , the primary function of circadian rhythms . Visual pigments are located in the brains of more complex organisms , and are thought to have a role in synchronising spawning with lunar cycles . By detecting the subtle changes in night - time illumination , organisms could synchronise the release of sperm and eggs to maximise the probability of fertilisation .
  Fact 0.17: Together with the leading flagellum, the euglena's eyespot allows the organism to move in response to light.

Retrieved items:
<item 1>
Evolution of the eye
This suggests that when the two lineages diverged in the Precambrian , they had only very primitive light receptors , which developed into more complex eyes
independently . Early eyes ( edit ) The basic light - processing unit of eyes is the photoreceptor cell , a specialized cell containing two types of molecules
in a membrane : the opsin , a light - sensitive protein , surrounding the chromophore , a pigment that distinguishes colors . Groups of such cells are termed ``
eyespots '' , and have evolved independently somewhere between 40 and 65 times . These eyespots permit animals to gain only a very basic sense of the direction
and intensity of light , but not enough to discriminate an object from its surroundings . Developing an optical system that can discriminate the direction of
light to within a few degrees is apparently much more difficult , and only six of the thirty - some phyla possess such a system . However , these phyla account
for 96 % of living species . The planarian has `` cup '' eyespots that can slightly distinguish light direction .
</item 1>

<item 2>
Evolution of the eye
Evolution of the eye - wikipedia Evolution of the eye Jump to : navigation , search Major stages in the evolution of the eye in vertebrates . The evolution of
the eye is attractive to study , because the eye distinctively exemplifies an analogous organ found in many animal forms . Complex , image - forming eyes have
evolved independently several times . The evolutionary geneticist Walter Gering holds a different opinion . According his theory , the genes engaged with
photoreceptor cells ( so called Pax 6 genes ) appeared probably at once in the life history , at a very early stage of the evolution . Probably , the complex ,
image forming kinds of eyes are at least five : Chidaria - lens eye , complex eye ; Molluscs - lens eye , everse retina ; Annelids - lens eye , everse retina ;
Arthropods - facet eye ; Vertebrate - lens eye , inverse retina Complex eyes appeared first within the few million years of the Cambrian explosion . From before
the Cambrian , no evidence of eyes has survived , but diverse eyes are known from the Burgess shale of the Middle Cambrian , and from the slightly older Emu Bay
Shale .
</item 2>

<item 3>
Evolution of the eye
PMID 7892602 . Jump up ^ Tomarev , S.I. ; Callaerts , P. ; Kos , L. ; Zinovieva , R. ; Halder , G. ; Gehring , W. ; Piatigorsky , J. ( 1997 ) . `` Squid Pax - 6
and eye development '' . Proceedings of the National Academy of Sciences of the United States of America . 94 ( 6 ) : 2421 -- 2426 . Bibcode : 1997PNAS ...
94.2421 T . doi : 10.1073 / pnas. 94.6. 2421 . PMC 20103 . PMID 9122210 . ^ Jump up to : Gehring , W.J. ( 13 January 2005 ) . `` New Perspectives on Eye
Development and the Evolution of Eyes and Photoreceptors '' ( Full text ) . Journal of Heredity . Oxford Journals. 96 ( 3 ) : 171 -- 184 .
</item 3>

<item 4>
Evolution of the eye
149 Jump up ^ Lee , M.S.Y. ; Jago , J.B. ; Garcia - Bellido , D.C. ; Edgecombe , G.E. ; Gehling , J. G ; Paterson , J.R. ( 2011 ) . `` Modern optics in
exceptionally preserved eyes of Early Cambrian arthropods from Australia '' . Nature . 474 : 631 -- 634 . doi : 10.1038 / nature10097 . PMID 21720369 . ^ Jump
up to : Darwin , Charles ( 1859 ) . On the Origin of Species . London : John Murray . Jump up ^ Gehring WJ ( 2005 ) . `` New perspectives on eye development and
the evolution of eyes and photoreceptors '' . J. Hered. 96 ( 3 ) : 171 -- 84 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Jump up ^ Nilsson , D. - E. (
2013 ) . `` Eye evolution and its functional basis '' . Visual Neuroscience.
</item 4>

<item 5>
Evolution of the eye
Jump up ^ Dawkins , Richard ( 1986 ) . The Blind Watchmaker . ^ Jump up to : Schoenemann , B. ; Liu , J.N. ; Shu , D.G. ; Han , J. ; Zhang , Z.F. ( 2008 ) . ``
A miniscule optimized visual system in the Lower Cambrian '' . Lethaia. 42 ( 3 ) : 265 -- 273 . doi : 10.1111 / j. 1502 - 3931.2008. 00138. x . Jump up ^ Ali ,
M.A. and M.A. Klyne. 1985 . Vision in vertebrates . New York : Plenum Press Jump up ^ Fernald , Russell D. ( 2001 ) . The Evolution of Eyes : Why Do We See What
We See ? Karger Gazette 64 : `` The Eye in Focus '' . Jump up ^ Fernald , Russell D. ( 1998 ) . Aquatic Adaptations in Fish Eyes . New York , Springer .
</item 5>

<item 6>
Evolution of the eye
Oxford Journals. 96 ( 3 ) : 171 -- 184 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Retrieved 2008 - 04 - 26 . ^ Jump up to : MF Land ; RD Fernald ( 1992
) . `` The Evolution of Eyes '' . Annual Review of Neuroscience. 15 : 1 -- 29 . doi : 10.1146 / annurev. ne. 15.030192. 000245 . PMID 1575438 . Jump up ^ Autrum
, H ( 1979 ) . `` Introduction '' . In H. Autrum . Comparative Physiology and Evolution of Vision in Invertebrates - A : Invertebrate Photoreceptors . Handbook
of Sensory Physiology . VII / 6A . New York : Springer - Verlag . pp. 6 -- 9 . ISBN 3 - 540 - 08837 - 7 .
</item 6>

<item 7>
Evolution of the eye
`` Eye evolution and its functional basis '' . Visual Neuroscience. 30 : 5 -- 20 . doi : 10.1017 / s0952523813000035 . ^ Jump up to : Nilsson , D. - E. ; Pelger
, S. ( 1994 ) . `` A pessimistic estimate of the time required for an eye to evolve '' . Proceedings of the Royal Society B : Biological Sciences . 256 : 53 --
58 . doi : 10.1098 / rspb. 1994.0048 . PMID 8008757 . Jump up ^ Young , G.C. ( 2008 ) . `` Early evolution of the vertebrate eye -- fossil evidence '' . Evo Edu
Outreach. 1 : 427 -- 438 . doi : 10.1007 / s12052 - 008 - 0087 - y . Jump up ^ Parker , A.R. ( 2009 ) . `` On the origin of optics '' . Optics & Laser
Technology . 43 ( 2 ) : 323 -- 329 .
</item 7>

<item 8>
Graph query row:
entity: eyespot
relationship: LOCATED_IN
related: anterior end
</item 8>

<item 9>
Evolution of the eye
1 ( 4 ) : 439 -- 447 . doi : 10.1007 / s12052 - 008 - 0084 - 1 . ISSN 1936 - 6426 . Further reading ( edit ) Lamb TD , Collin SP , Pugh EN ( December 2007 ) .
`` Evolution of the vertebrate eye : opsins , photoreceptors , retina and eye cup '' . Nat. Rev. Neurosci. 8 ( 12 ) : 960 -- 76 . doi : 10.1038 / nrn2283 . PMC
3143066 . PMID 18026166 . Illustration . Review Lamb , TD ( 2011 ) . `` Evolution of the Eye '' ( PDF ) . Scientific American . 305 : 64 -- 69 . doi : 10.1038 /
scientificamerican0711 - 64 . Land , Michael F. ; Nilsson , Dan - Eric ( 2012 ) . `` The origin of vision '' . Animal Eyes ( 2 ed . ) . Oxford : Oxford
University Press .
</item 9>

<item 10>
Evolution of the eye
PMID 25734540 . Greuet , C ( 1968 ) . `` Organisation ultrastructurale de l'ocelle de deux Peridiniens Warnowiidae , Erythropsis pavillardi Kofoid et Swezy et
Warnowia pulchra Schiller '' . Protistologica. 4 : 209 -- 230 . Gregory S. Gavelis , Shiho Hayakawa , Richard A. White III , Takashi Gojobori , Curtis A. Suttle
, Patrick J. Keeling , Brian S. Leander ( 2015 ) . `` Eye - like ocelloids are built from different endosymbiotically acquired components '' . Nature . 523 :
204 -- 7 . Bibcode : 2015Natur. 523 ... 204G . doi : 10.1038 / nature14593 . PMID 26131935 . CS1 maint : Uses authors parameter ( link ) Oakley , Todd H. ;
Speiser , Daniel I. ( 2015 ) . `` How Complexity Originates : The Evolution of Animal Eyes '' .
</item 10>
=== END ITEM id=a192-0196e65757df2a04 ===

=== ITEM id=a151-0bee757f6d93d612 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Italian Title / RAI English Title 4Kids English title Italian air date American air date `` Una Fata a Gardenia '' / `` An Unexpected Event '' `` It Feels Like Magic '' 28 January 2004 ( 2004 - 01 - 28 ) 19 June 2004 While on summer vacation , Bloom , an ordinary 16 - year - old high school girl living on Earth , encounters a battle between a 17 - year - old fairy named Stella and an ogre named Knut . As Stella 's magic scepter is stolen , Bloom intervenes , discovering that she has magical powers and manages to turn the tide . Stella invites Bloom to attend Alfea , a school for fairies , to fully master her new powers . Knut returns with more ghouls and a troll , launching a surprise attack at Bloom 's home . Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky , Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School '' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents , she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy , Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her . Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school , Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded . She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball . Later Bloom spots the Trix trying to steal Stella 's ring , and tries to fight them off , transforming in a fairy for the first time . Although the Trix are able to escape , the other Winx girls inform Bloom that Stella 's ring is safe and that they swapped the contents of the case with a baby duck ( Pepe ) that bonds with Icy . `` La palude di Melmamora '' / `` The Black - Mud Swamp '' `` The Voice of Nature '' 4 February 2004 ( 2004 - 02 - 04 ) 10 July 2004 As part of Professor Palladium 's field trip , the class must navigate the Black - Mud Swamp to reach a clearing in the Gloomy - Wood Forest by listening to nature and without using any magic . The boys from Red Fountain are transporting a troll prisoner but the Trix cause their aircraft to crash land , and the troll is freed . The Winx and the Red Fountain boys end up meeting in the swamp , and must work together , despite their differences , to find their way out . They encounter the troll who is harassing some other Alfea fairies , and chase him off , however , the Trix find the troll and send him away . Although the Red Fountain boys failed to deliver the troll , they remark they are starting to feel like a team . The Winx girls , although they finished last in their assignment , are commended by their teachers for their helping others . 5 5 `` Appuntamento al buio '' / `` Date with Disaster '' `` Date with Disaster '' 6 February 2004 ( 2004 - 02 - 06 ) 17 July 2004 While the Winx are on kitchen duty , Stella is invited on a date with Prince Sky ( Brandon ) at the brand - new Black Lagoon restaurant and also asks for help on picking out an outfit . While selecting her attire , Stella decides not to wear the ring of Solaria and gives it to Bloom for safekeeping ( believing she wo n't need her magic powers ) . The next morning , Stella starts acting completely different , becoming irritable and refusing to say antything about last night . During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria ! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower '' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out what made the Trix go to such lengths to get it . Meanwhile , the Trix use a spell on the ring , hoping to access an extremely powerful magic known as the Dragon Flame , only to have their spell blow up in their faces as the ring proves negative ! An infuriated Icy throws the ring away and plans to begin the search all over again . Back at Alfea , Bloom learns more on the ring of Solaria , but also learns about the Dragon Flame . Bloom then turns to Faragonda for answers , who tells her that the Dragon Flame was the energy of the Great Dragon , who created the magical universe and later rested on the planet Domino . That night , the Winx plan to infiltrate Cloud Tower to retrieve Stella 's ring ( since she ca n't transform without it ) . Using the same secret tunnel the Trix used earlier , the girls sneak into Cloudtower without a hitch while the Trix are out . After searching their room , Stella finds her ring , but the tunnel does n't reopen and the Winx are forced to find the main exit without getting caught . While exploring , the Winx stumble upon Cloudtower 's archive which contains history on every person in the Magic dimension . A curious Bloom spots a book with her name on it and tries to read more on her past , but quickly regrets it when it turns into a trap that alerts Headmistress Griffin ! The Winx try to make a hasty escape , but Griffin sends Bako monsters to scare them out ! After using their magic to fight the monsters , the Winx are led to a dead end . In a panic to destroy the last monster , Stella accidentally sets the room on fire ! As the Winx begin to suffocate , a mysterious voice guides Bloom to an exit , allowing them to leave Cloudtower . Though the ring was recovered , the Winx are busted since Griffin already called Faragonda hours ago . As punishment , Faragonda temporarily extracts all their powers . 7 7 `` A che servono gli amici ? '' / `` Friends in Need '' `` Grounded '' 11 February 2004 ( 2004 - 02 - 11 ) 31 July 2004 The Winx have lost all their magic as punishment for sneaking into Cloudtower , but to make matters worse , Griselda tasks them with cleaning the entire school without their magic ( or technology , much to Tecna 's dismay ) while the rest of the students and faculty go to a concert . Problems continue as Tecna struggles with no gadgets to help her , causing her to wear a bucket on her head and attempt to use a broom , while Musa , Stella and Bloom have a water fight , wasting their time , energy and supplies . In order to make the punishment lighter , the girls secretly invite the boys from Redfountain over after everyone has left , claiming to be throwing an all - night party . Meanwhile , the Trix use the opportunity to sneak into a near - empty Alfea to find the real source of the Dragon Flame . The boys arrive and instead learn they 'll be working as janitors , though they still play music as they work . Soon after , the Trix teleport to Alfea and summon the vacuums to guide them towards the Dragon Flame , while distracting the gang by summoning a monster . The Winx and specialists quickly meet their distraction , and while the boys try to fight with their weapons , their quickly knocked aside . The Winx , despite having no magic , manage to outsmart the beast by using floor cleaner . Unable to find the Dragon Flame , the Trix depart to Cloudtower , unseen by anyone while the Specialists head back to Redfountain . The next morning , Faragonda sees the damage and hears the story from the Winx and as a reward for defeating an adversary using only wits and resourcefulness , the Winx 's powers are returned to them . 8 8 `` La festa della rosa '' / `` A Friendship Sundered '' `` The Day of the Rose '' 13 February 2004 ( 2004 - 02 - 13 ) 7 August 2004 During the Rose Festival , both Bloom and Riven unwittingly fall into a trap by the Trix to divide the close relationship between the Specialists and the Winx . In the end , Riven thinks that Darcy has saved him in a racing crash and he blames Bloom for what happened then gets mad at his friends for believing Bloom over him and transfers out of the squad of Specialists along with Darcy . 9 9 `` Il tradimento di Riven '' / `` Betrayed ! '' `` Spelled '' 16 February 2004 ( 2004 - 02 - 16 ) 14 August 2004 There are hostilities between the Winx due to Stella 's behavior , and Musa 's feelings for Riven are strained . In a fight with the Trix , the possessor of the Dragon 's Flame is revealed to be Bloom . 10 10 `` La fiamma del drago '' / `` Bloom Tested '' `` Magical Reality Check '' 18 February 2004 ( 2004 - 02 - 18 ) 21 August 2004 It 's time for Bloom 's magical midterm test in a virtual reality machine . The Trix mess with the computer so they can enter it and test her to be sure of Bloom is indeed the one who possesses the Dragon Flame . 11 11 `` Il regno delle ninfe '' / `` The Monster and The Willow '' `` Junior League '' 20 February 2004 ( 2004 - 02 - 20 ) 28 August 2004 Flora experiments with her plants to study for her test . Stella ( who had her clothes torn apart by a plant ) and Tecna ( who was attacked by a plant ) tell Flora to move the plants . She moves them to Black - Mud Swamp . Flora seeks help from the Water Nymphs in finding a special plant . The Water Nymphs tell the Winx about a monster which has been terrorizing them and the Winx decide to fight it . 12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom , which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13 13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop . A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 - 02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March 2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03 ) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17 `` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! '' / `` The Font of Dragon Fire '' `` Senior Witches Go to Earth '' 8 March 2004 ( 2004 - 03 - 08 ) 16 October 2004 Upset about the events at Redfountain , Bloom returns to Earth . The Trix take advantage of the situation and attack her on Earth . The witches reveal to Bloom that she is the surviving heir to planet Domino ( Sparks ) , and they steal the Dragon Fire from her . 19 19 `` Attacco ad Alfea '' / `` The Fall of Magix '' `` The Army of Decay '' 10 March 2004 ( 2004 - 03 - 10 ) 23 October 2004 Bloom returns to Alfea following the events on Earth . The Trix take control of Cloud Tower and summon the Army of Decay to take control of Magix . 20 20 `` La scomparsa di Bloom '' / `` Mission to Domino '' `` Sparks of Hope '' 12 March 2004 ( 2004 - 03 - 12 ) 30 October 2004 To counter the powerful magic possessed by the Trix , the Winx Club go to planet Domino hoping to find the Dragon 's Flame . Redfountain is attacked by the Army of Decay . 21 21 `` Trappola di ghiaccio '' / `` The Crown of Dreams '' `` The Frozen Palace '' 15 March 2004 ( 2004 - 03 - 15 ) 6 November 2004 In the ruins of Domino ( Sparks ) , the Winx are attacked during their exploration . In Magix , Redfountain is destroyed by the Trix . 22 22 `` Il ritorno di Riven '' / `` Storming Cloudtower '' `` Mission to Cloud Tower '' 17 March 2004 ( 2004 - 03 - 17 ) 13 November 2004 The Army of Decay forces the retreat of the population to Alfea . The Winx go to Cloud Tower in attempt to retrieve the power of the Dragon 's Flame from the Trix . Riven rejoins the Specialists . 23 23 `` Fuga da Torrenuvola '' / `` Power Play '' `` The Search for the Flame '' 19 March 2004 ( 2004 - 03 - 19 ) 20 November 2004 The Trix torment the Winx for being unable to find the power of the Dragon 's Flame on Domino ( Sparks ) . The cadre and staff of Cloud Tower escape to Alfea . 24 24 `` Il mistero del lago '' / `` The Witches ' Siege '' `` Battle for Alfea '' 22 March 2004 ( 2004 - 03 - 22 ) 27 November 2004 The Army of Darkness launches an attack on Alfea . Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice . 25 25 `` Il sonno di Magix '' / `` The Ultimate Challenge '' `` The Great Witch Invasion '' 24 March 2004 ( 2004 - 03 - 24 ) 4 December 2004 The first attack on Alfea fails . Bloom follows the spirit of her older - sister Daphne and regains her Dragon Fire powers . The Trix lead the Army of Decay in a final attack on Alfea . 26 26 `` Battaglia finale '' / `` The Witches ' Downfall '' `` Fire and Ice '' 26 March 2004 ( 2004 - 03 - 26 ) 11 December 2004 As the Army of Darkness approaching Alfea and the Trix seeking to take complete control of the Magix Dimension , the Winx , except Bloom and including Alfea 's students and teachers take action to fend off the Trix from completely destroying algae to the ground before the Trix use the powers of the Dragon . RC1 RC1 `` Attack of the Senior Witches '' N / A 22 May 2004 A clip show episode made by 4Kids Entertainment featuring scenes from episodes 18 , 19 and 20 where the Trix attack Alfea , serving as a sneak preview to the first season .
  Fact 0.63: The Trix use the opportunity to sneak into a near-empty Alfea to find the real source of the Dragon Flame.

Retrieved items:
<item 1>
Winx Club (season 1)
A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning
Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 -
02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower
with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March
2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test .
</item 1>

<item 2>
Winx Club (season 1)
Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head
Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello
di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school ,
Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded
. She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix
casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball .
</item 2>

<item 3>
Winx Club (season 1)
The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria
! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic
energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms
out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower
'' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out
what made the Trix go to such lengths to get it .
</item 3>

<item 4>
Winx Club (season 1)
12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud
Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other
contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom
, which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13
13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth
during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop .
</item 4>

<item 5>
Winx Club (season 1)
Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky ,
Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School
'' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her
admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents ,
she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy ,
Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her .
</item 5>

<item 6>
Winx Club (season 1)
10 10 `` La fiamma del drago '' / `` Bloom Tested '' `` Magical Reality Check '' 18 February 2004 ( 2004 - 02 - 18 ) 21 August 2004 It 's time for Bloom 's
magical midterm test in a virtual reality machine . The Trix mess with the computer so they can enter it and test her to be sure of Bloom is indeed the one who
possesses the Dragon Flame . 11 11 `` Il regno delle ninfe '' / `` The Monster and The Willow '' `` Junior League '' 20 February 2004 ( 2004 - 02 - 20 ) 28
August 2004 Flora experiments with her plants to study for her test . Stella ( who had her clothes torn apart by a plant ) and Tecna ( who was attacked by a
plant ) tell Flora to move the plants . She moves them to Black - Mud Swamp . Flora seeks help from the Water Nymphs in finding a special plant . The Water
Nymphs tell the Winx about a monster which has been terrorizing them and the Winx decide to fight it .
</item 6>

<item 7>
List of Marvel Comics characters: B
It is unknown how long she remained in that state , and whether she retained or lost her mutant abilities after the M - Day . Bloom 's legacy however , was to
leave a lasting impression upon Emma that baseline humans ( and even non-telepathic mutants ) were little more than `` soulless meat puppets '' , who existed
for the sole purpose of the amusement of those capable of controlling them . The Emma Frost comic series ends with an ambulance carrying the comatose Astrid
into the distance ; with Emma staring on , while casually listening to the background thoughts of others in attendance , and thinking that Astrid was right
about all non-telepaths after all . Had the series continued , it was this event that would have eventually set in motion the events that would see Emma rise to
power as both the head of Frost Industries , and as White Queen of the Hellfire club . Astrid Bloom is a mutant with various telepathic abilities . She can read
minds and project her thoughts into the minds of others .
</item 7>

<item 8>
Winx Club (season 1)
Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice . 25 25 `` Il sonno di Magix '' / `` The Ultimate Challenge '' `` The Great Witch
Invasion '' 24 March 2004 ( 2004 - 03 - 24 ) 4 December 2004 The first attack on Alfea fails . Bloom follows the spirit of her older - sister Daphne and regains
her Dragon Fire powers . The Trix lead the Army of Decay in a final attack on Alfea . 26 26 `` Battaglia finale '' / `` The Witches ' Downfall '' `` Fire and
Ice '' 26 March 2004 ( 2004 - 03 - 26 ) 11 December 2004 As the Army of Darkness approaching Alfea and the Trix seeking to take complete control of the Magix
Dimension , the Winx , except Bloom and including Alfea 's students and teachers take action to fend off the Trix from completely destroying algae to the ground
before the Trix use the powers of the Dragon .
</item 8>

<item 9>
Winx Club (season 1)
Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03
) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17
`` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at
Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon
switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! ''
</item 9>

<item 10>
Winx Club (season 1)
During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella
begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince
Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the
Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives
them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She
quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls
transform , and quickly learn that the entire date was a plot by the Trix .
</item 10>
=== END ITEM id=a151-0bee757f6d93d612 ===

=== ITEM id=a181-11494f8a6120a450 ===
Gold sentences and facts:
Sentence 0: Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name ( Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741 7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743 -- First Lord of the Treasury Whig Carteret 7002502000000000000 ♠ 1 year and 137 days ​ The Right Honourable Henry Pelham FRS MP for Sussex ( 1694 -- 1754 ) 27 August 1743 6 March 1754 1747 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Broad Bottom I Broad Bottom II 7003384500000000000 ♠ 10 years and 192 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 16 March 1754 16 November 1756 1754 First Lord of the Treasury Leader of the House of Lords Whig Newcastle I 7002977000000000000 ♠ 2 years and 246 days His Grace William Cavendish 4th Duke of Devonshire KG PC ( 1720 -- 1764 ) 16 November 1756 25 June 1757 -- First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Devonshire ​ 1757 Caretaker 7002222000000000000 ♠ 222 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 2 July 1757 26 May 1762 1761 First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Newcastle George III ( 1760 -- 1820 ) 7003179000000000000 ♠ 4 years and 329 days The Right Honourable John Stuart 3rd Earl of Bute KG PC ( 1713 -- 1792 ) 26 May 1762 8 April 1763 -- First Lord of the Treasury Leader of the House of Lords Tory Bute ( Tory -- Whig ) 7002318000000000000 ♠ 318 days The Right Honourable George Grenville MP for Buckingham ( 1712 -- 1770 ) 16 April 1763 13 July 1765 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig ( Grenvillite ) Grenville ( Whig -- Tory ) 7002820000000000000 ♠ 2 years and 89 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 13 July 1765 30 July 1766 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham I 7002383000000000000 ♠ 1 year and 18 days The Right Honourable William Pitt 1st Earl of Chatham PC FRS MP for Bath ( 1708 -- 1778 ) 30 July 1766 14 October 1768 1768 Lord Keeper of the Privy Seal Whig ( Chathamite ) Chatham ( Whig -- Tory ) 7002808000000000000 ♠ 2 years and 77 days His Grace Augustus FitzRoy 3rd Duke of Grafton KG PC ( 1735 -- 1811 ) 14 October 1768 28 January 1770 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Grafton 7002472000000000000 ♠ 1 year and 107 days The Right Honourable Frederick North Lord North KG MP for Banbury ( 1732 -- 1792 ) 28 January 1770 22 March 1782 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Northite ) North 1774 1780 7003443700000000000 ♠ 12 years and 54 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 27 March 1782 1 July 1782 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham II 7001970000000000000 ♠ 97 days The Right Honourable William Petty 2nd Earl of Shelburne KG PC ( 1737 -- 1805 ) 4 July 1782 2 April 1783 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Shelburne ( Whig -- Tory ) 7002273000000000000 ♠ 273 days His Grace William Cavendish - Bentinck 3rd Duke of Portland PC FRS ( 1738 -- 1809 ) 2 April 1783 19 December 1783 -- First Lord of the Treasury Leader of the House of Lords Whig Fox -- North 7002262000000000000 ♠ 262 days The Right Honourable William Pitt the Younger MP for Appleby -- Cambridge University ( 1759 -- 1806 ) 19 December 1783 14 March 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt I 1784 1790 1796 7003629500000000000 ♠ 17 years and 86 days The Right Honourable Henry Addington MP for Devizes ( 1757 -- 1844 ) 17 March 1801 10 May 1804 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Addington 1802 7003115100000000000 ♠ 3 years and 55 days The Right Honourable William Pitt the Younger MP for Cambridge University ( 1759 -- 1806 ) 10 May 1804 23 January 1806 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt II 7002624000000000000 ♠ 1 year and 259 days The Right Honourable William Grenville 1st Baron Grenville PC PC ( Ire ) ( 1759 -- 1834 ) 11 February 1806 31 March 1807 First Lord of the Treasury Leader of the House of Lords Whig All the Talents ( Whig -- Tory ) 1806 7002414000000000000 ♠ 1 year and 49 days His Grace William Cavendish - Bentinck 3rd Duke of Portland KG PC FRS ( 1738 -- 1809 ) 31 March 1807 4 October 1809 First Lord of the Treasury Tory ( Pittite ) Portland II 1807 7002919000000000000 ♠ 2 years and 188 days The Right Honourable Spencer Perceval KC MP for Northampton ( 1762 -- 1812 ) 4 October 1809 11 May 1812 -- First Lord of the Treasury Chancellor of the Exchequer Chancellor of the Duchy of Lancaster Leader of the House of Commons Tory ( Pittite ) Perceval 7002951000000000000 ♠ 2 years and 221 days The Right Honourable Robert Jenkinson 2nd Earl of Liverpool KG PC FRS ( 1770 -- 1828 ) 8 June 1812 9 April 1827 1812 First Lord of the Treasury Leader of the House of Lords Tory ( Pittite ) Liverpool 1818 1820 George IV ( 1820 -- 1830 ) 1826 7003541900000000000 ♠ 14 years and 306 days The Right Honourable George Canning FRS MP for Seaford ( 1770 -- 1827 ) 10 April 1827 8 August 1827 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Canningite ) Canning ( Canningite -- Whig ) 7002121000000000000 ♠ 121 days The Right Honourable Frederick John Robinson 1st Viscount Goderich PC ( 1782 -- 1859 ) 31 August 1827 21 January 1828 -- First Lord of the Treasury Leader of the House of Lords Tory ( Canningite ) Goderich 7002144000000000000 ♠ 144 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 22 January 1828 16 November 1830 First Lord of the Treasury Leader of the House of Lords Tory Wellington -- Peel 1830 William IV ( 1830 -- 1837 ) 7003103000000000000 ♠ 2 years and 299 days The Right Honourable Charles Grey 2nd Earl Grey KG PC ( 1764 -- 1845 ) 22 November 1830 9 July 1834 First Lord of the Treasury Leader of the House of Lords Whig Grey 1831 1832 7003132600000000000 ♠ 3 years and 230 days The Right Honourable William Lamb 2nd Viscount Melbourne PC ( 1779 -- 1848 ) 16 July 1834 14 November 1834 -- First Lord of the Treasury Leader of the House of Lords Whig Melbourne I 7002122000000000000 ♠ 122 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 14 November 1834 10 December 1834 -- First Lord of the Treasury Sec. of State for the Home Department Sec. of State for Foreign Affairs Sec. of State for War and the Colonies Leader of the House of Lords Tory Wellington Caretaker 7001270000000000000 ♠ 27 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 10 December 1834 8 April 1835 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Conservative Peel I 7002120000000000000 ♠ 120 days The Right Honourable William Lamb 2nd Viscount Melbourne PC FRS ( 1779 -- 1848 ) 18 April 1835 30 August 1841 1835 First Lord of the Treasury Leader of the House of Lords Whig Melbourne II 1837 Victoria ( 1837 -- 1901 ) 7003232700000000000 ♠ 6 years and 135 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 30 August 1841 29 June 1846 1841 First Lord of the Treasury Leader of the House of Commons Conservative Peel II 7003176500000000000 ♠ 4 years and 304 days The Right Honourable Lord John Russell FRS MP for the City of London ( 1792 -- 1878 ) 30 June 1846 21 February 1852 1847 First Lord of the Treasury Leader of the House of Commons Whig Russell I 7003206300000000000 ♠ 5 years and 237 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 23 February 1852 17 December 1852 First Lord of the Treasury Leader of the House of Lords Conservative Who ? Who ? 1852 7002299000000000000 ♠ 299 days The Right Honourable George Hamilton - Gordon 4th Earl of Aberdeen KT FRSE FRS PC FSA Scot ( 1784 -- 1860 ) 19 December 1852 30 January 1855 -- First Lord of the Treasury Leader of the House of Lords Peelite Aberdeen ( Peelite -- Whig ) 7002773000000000000 ♠ 2 years and 43 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 6 February 1855 19 February 1858 First Lord of the Treasury Leader of the House of Commons Whig Palmerston I 1857 7003111000000000000 ♠ 3 years and 14 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 20 February 1858 11 June 1859 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli II 7002477000000000000 ♠ 1 year and 112 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 12 June 1859 18 October 1865 1859 First Lord of the Treasury Leader of the House of Commons Liberal Palmerston II 1865 7003232100000000000 ♠ 6 years and 129 days The Right Honourable John Russell 1st Earl Russell KG PC FRS ( 1792 -- 1878 ) 29 October 1865 26 June 1866 -- First Lord of the Treasury Leader of the House of Lords Liberal Russell II 7002241000000000000 ♠ 241 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby KG PC ( 1799 -- 1869 ) 28 June 1866 25 February 1868 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli III 7002608000000000000 ♠ 1 year and 243 days The Right Honourable Benjamin Disraeli MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 -- 7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and 77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April 1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone II 23 April 1880 9 June 1885 1880 7003187400000000000 ♠ 5 years and 48 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 23 June 1885 28 January 1886 -- Sec. of State for Foreign Affairs Leader of the House of Lords Conservative Salisbury I 7002220000000000000 ♠ 220 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone III 1 February 1886 20 July 1886 1885 7002170000000000000 ♠ 170 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 July 1886 11 August 1892 1886 Leader of the House of Lords First Lord of the Treasury Sec. of State for Foreign Affairs Conservative Salisbury II 7003221000000000000 ♠ 6 years and 18 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone IV 15 August 1892 2 March 1894 1892 7002565000000000000 ♠ 1 year and 200 days The Right Honourable Archibald Primrose 5th Earl of Rosebery KG PC FRS ( 1847 -- 1929 ) 5 March 1894 22 June 1895 -- First Lord of the Treasury Lord President of the Council Leader of the House of Lords Liberal Rosebery 7002475000000000000 ♠ 1 year and 110 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 June 1895 11 July 1902 Leader of the House of Lords Sec. of State for Foreign Affairs Lord Keeper of the Privy Seal Conservative Salisbury III ( Con. -- Lib. U . ) 1895 1900 7003257300000000000 ♠ 7 years and 17 days Edward VII ( 1901 -- 1910 ) The Right Honourable Arthur Balfour FRS DL MP for Manchester East ( 1848 -- 1930 ) 11 July 1902 5 December 1905 -- First Lord of the Treasury Leader of the House of Commons Conservative Balfour 7003124400000000000 ♠ 3 years and 148 days The Right Honourable Sir Henry Campbell - Bannerman GCB MP for Stirling Burghs ( 1836 -- 1908 ) 5 December 1905 7 April 1908 First Lord of the Treasury Leader of the House of Commons Liberal Campbell - Bannerman 1906 7002855000000000000 ♠ 2 years and 125 days The Right Honourable Herbert Henry Asquith KC FRS MP for East Fife ( 1852 -- 1928 ) 7 April 1908 5 December 1916 First Lord of the Treasury Leader of the House of Commons Sec. of State for War Liberal Asquith I 1910 ​ 1910 George V ( 1910 -- 1936 ) ​ Asquith II ( Lib. -- Con. -- et al . ) 7003316500000000000 ♠ 8 years and 243 days The Right Honourable David Lloyd George OM MP for Caernarvon Boroughs ( 1863 -- 1945 ) 6 December 1916 19 October 1922 First Lord of the Treasury Liberal Lloyd George War 1918 Lloyd George II ( Lib. -- Con . ) 7003214400000000000 ♠ 5 years and 318 days The Right Honourable Bonar Law MP for Glasgow Central ( 1858 -- 1923 ) 23 October 1922 20 May 1923 1922 First Lord of the Treasury Leader of the House of Commons Conservative ( Scot. U . ) Law 7002210000000000000 ♠ 210 days The Right Honourable Stanley Baldwin JP MP for Bewdley ( 1867 -- 1947 ) 23 May 1923 16 January 1924 -- First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Conservative Baldwin I 7002239000000000000 ♠ 239 days The Right Honourable Ramsay MacDonald MP for Aberavon ( 1866 -- 1937 ) 22 January 1924 4 November 1924 1923 First Lord of the Treasury Leader of the House of Commons Sec. of State for Foreign Affairs Labour MacDonald I 7002288000000000000 ♠ 288 days The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 4 November 1924 5 June 1929 1924 First Lord of the Treasury Leader of the House of Commons Conservative Baldwin II 7003167500000000000 ♠ 4 years and 214 days The Right Honourable Ramsay MacDonald FRS MP for Seaham ( 1866 -- 1937 ) 5 June 1929 7 June 1935 1929 First Lord of the Treasury Leader of the House of Commons Labour MacDonald II National Labour National I ( N. Lab. -- Con. -- et al . ) 1931 National II 7003219400000000000 ♠ 6 years and 3 days ​ The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 7 June 1935 28 May 1937 1935 First Lord of the Treasury Leader of the House of Commons Conservative National III ( Con. -- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​ Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26 July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker ( Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26 July 1945 26 October 1951 1945 First Lord of the Treasury Minister of Defence Labour Attlee I 1950 Attlee II 7003228400000000000 ♠ 6 years and 93 days The Right Honourable Sir Winston Churchill KG OM CH TD PCc DL FRS RA MP for Woodford ( 1874 -- 1965 ) 26 October 1951 6 April 1955 1951 First Lord of the Treasury Minister of Defence Conservative Churchill III 7003125900000000000 ♠ 3 years and 163 days Elizabeth II ( 1952 -- present ) The Right Honourable Sir Anthony Eden KG MC MP for Warwick & Leamington ( 1897 -- 1977 ) 6 April 1955 10 January 1957 1955 First Lord of the Treasury Conservative Eden 7002646000000000000 ♠ 1 year and 280 days The Right Honourable Harold Macmillan FRS MP for Bromley ( 1894 -- 1986 ) 10 January 1957 19 October 1963 1959 First Lord of the Treasury Conservative Macmillan I ​ Macmillan II 7003247400000000000 ♠ 6 years and 283 days The Right Honourable Sir Alec Douglas - Home KT MP for Kinross & Western Perthshire ( 1903 -- 1995 ) 19 October 1963 16 October 1964 -- First Lord of the Treasury Conservative ( Scot. U . ) Douglas - Home 7002364000000000000 ♠ 364 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 16 October 1964 19 June 1970 1964 First Lord of the Treasury Minister for the Civil Service Labour Wilson I 1966 Wilson II 7003207300000000000 ♠ 5 years and 247 days The Right Honourable Edward Heath MBE MP for Bexley -- Sidcup ( 1916 -- 2005 ) 19 June 1970 4 March 1970 First Lord of the Treasury Minister for the Civil Service Conservative Heath 7003135500000000000 ♠ 3 years and 259 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 4 March 5 April 1976 First Lord of the Treasury Minister for the Civil Service Labour Wilson III Wilson IV 7002764000000000000 ♠ 2 years and 33 days The Right Honourable James Callaghan MP for Cardiff South East ( 1912 -- 2005 ) 5 April 1976 4 May 1979 -- First Lord of the Treasury Minister for the Civil Service Labour Callaghan 7003112500000000000 ♠ 3 years and 30 days The Right Honourable Margaret Thatcher FRS FRIC MP for Finchley ( 1925 -- 2013 ) See also § Premierships :3 First Lord of the Treasury Minister for the Civil Service Conservative Thatcher I 4 May 1979 28 November 1990 1979 Thatcher II Thatcher III 7003422700000000000 ♠ 11 years and 209 days The Right Honourable John Major MP for Huntingdon ( born 1943 ) 28 November 1990 2 May 1997 First Lord of the Treasury Minister for the Civil Service Conservative Major I ​ Major II 7003234800000000000 ♠ 6 years and 156 days The Right Honourable Tony Blair MP for Sedgefield ( born 1953 ) See also § Premierships : 4 First Lord of the Treasury Minister for the Civil Service Labour Blair I 2 May 1997 27 June 2007 1997 Blair II 2005 Blair III 7003370900000000000 ♠ 10 years and 57 days The Right Honourable Gordon Brown MP for Kirkcaldy & Cowdenbeath ( born 1951 ) See also § Premierships : 5 First Lord of the Treasury Minister for the Civil Service Labour Brown 27 June 2007 11 May -- 7003105000000000000 ♠ 2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July 2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days
  Fact 0.48: Henry Addington was Prime Minister from 17 March 1801 to 10 May 1804, for 3 years and 55 days.

Retrieved items:
<item 1>
List of Prime Ministers of the United Kingdom
`` Boroughs to Lose One Member '' . Parliamentary Debates ( Hansard ) . 296 . House of Commons . col. 156 -- 157 . Jump up ^ Sir Donald Horne Macfarlane ( 14
April 1885 ) . `` Central Asia -- Russia and Afghanistan -- the Russo -- Afghan Frontier -- Russian Advances '' . Parliamentary Debates ( Hansard ) . 296 .
House of Commons . col. 1634 -- 1635 . ^ Jump up to : Marriott ( 1923 ) , p. 83 . Jump up ^ Clarke ( 1999 ) , p. 266 . Jump up ^ Hennessy ( 2001 ) , p. 39 .
Jump up ^ `` Parties and Prime Ministers '' . BBC News . 19 May 1998 . Archived from the original on 18 March 2004 . Retrieved 12 October 2008 . Jump up ^
Mackay , Robert ( 28 December 1987 ) . `` Thatcher longest serving British prime minister '' . United Press International .
</item 1>

<item 2>
List of Prime Ministers of the United Kingdom
British Prime Ministers . Amberley Publishing Limited . ISBN 978 - 1 - 4456 - 1242 - 3 . Quinault , Roland ( 17 March 2011 ) . British Prime Ministers and
Democracy : From Disraeli to Blair . Bloomsbury Publishing . ISBN 978 - 1 - 4411 - 1105 - 0 . External links ( edit ) `` Past Prime Ministers '' . gov.uk .
Government of the United Kingdom . Archived from the original on 25 August 2008 . `` Prime Ministers and Politics Timeline '' . BBC History . Archived from the
original on 27 May 2011 . ( hide ) Lists relating to Prime Ministers of the United Kingdom List of Prime Ministers of the United Kingdom Lists by order Age
Constituencies Cultural depictions Governments Historical rankings Number living Records Service ( timeline ) Term length Peerages By monarch Victoria Edward
VII George V Edward VIII George VI Elizabeth II Personal life Burial places Coats of arms Education Nickname Spouse In fiction Fictional Prime Ministers
Biography portal British politics portal Politics portal Lists portal Retrieved from `` https://en.wikipedia.
</item 2>

<item 3>
List of Prime Ministers of the United Kingdom
wikipedia.org/w/index.php?title=List_of_Prime_Ministers_of_the_United_Kingdom&oldid=821003615 '' Categories : Lists of Prime Ministers of the United Kingdom
Lists of government ministers of the United Kingdom Lists of prime ministers Prime Ministers of the United Kingdom Hidden categories : Wikipedia articles
incorporating a citation from the ODNB Wikipedia articles incorporating a citation from the 1911 Encyclopaedia Britannica with Wikisource reference Wikipedia
articles incorporating text from the 1911 Encyclopædia Britannica Articles incorporating Cite DNB template Pages using div col without cols and colwidth
parameters Featured lists Use dmy dates from September 2011 Talk Contents About Wikipedia Беларуская Čeština Cymraeg Dansk Deutsch Eesti Ελληνικά Español
Esperanto فارسی Français Frysk Gàidhlig 한국어 Հայերեն Hrvatski Ido Bahasa Indonesia Íslenska ქართული Latina Latviešu
</item 3>

<item 4>
List of Prime Ministers of the United Kingdom
BBC News . 13 July 2016 . Retrieved 26 July 2016 . Lay summary ( 13 July 2016 ) . Bibliography ( edit ) Burt , Llewellyn C. ( 1874 ) . A synoptical history of
England ( 2nd ed . ) . London : Lockwood . Carpenter , Clive , ed. ( 1992 ) . The Guinness UK Data Book . Guinness . ISBN 978 - 0 - 85112 - 522 - 0 . Chapman ,
Richard A. ( 31 January 2002 ) . `` History : from earliest times to the present day '' . The Treasury in Public Policy - Making . Routledge . ISBN 978 - 1 -
134 - 86426 - 3 . Clarke , John ( 1999 ) . `` House of Hanover '' . In Fraser , Antonia . The Lives of the Kings & Queens of England . University of California
Press ( published February 2000 ) .
</item 4>

<item 5>
List of Prime Ministers of the United Kingdom
Davidson , Jonathan ( 30 November 2010 ) . Downing Street Blues : A History of Depression and Other Mental Afflictions in British Prime Ministers . McFarland .
ISBN 978 - 0 - 7864 - 5793 - 9 . Grube , Dennis ( 11 October 2013 ) . Prime Ministers and Rhetorical Governance . Palgrave Macmillan UK . ISBN 978 - 1 - 137 -
31836 - 7 . King , Anthony Stephen , ed. ( 1985 ) . The British Prime Minister ( 2nd ed . ) . Duke University Press . ISBN 0 - 8223 - 0635 - 2 . Leonard , Dick
( 10 July 2008 ) . Nineteenth Century Premiers : Pitt to Rosebery . Palgrave Macmillan UK . ISBN 978 - 0 - 230 - 22725 - 5 . Parker , Robert J. ( 15 March 2011
) . British Prime Ministers . Amberley Publishing Limited .
</item 5>

<item 6>
List of Prime Ministers of the United Kingdom
-- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 --
7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the
Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and
77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord
of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April
1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William
</item 6>

<item 7>
List of Prime Ministers of the United Kingdom
List of Prime Ministers of the United Kingdom - wikipedia List of Prime Ministers of the United Kingdom Jump to : navigation , search Top left : Walpole became
the first Prime Minister of the Kingdom of Great Britain in 1721 . Top right : Pitt the Younger became the first Prime Minister of the United Kingdom of Great
Britain and Ireland in 1801 . Bottom left : Lloyd George was the last Prime Minister of the United Kingdom of Great Britain and Ireland as a result of the Anglo
- Irish Treaty in 1922 . Bottom right : Baldwin was Prime Minister when the country was renamed the United Kingdom of Great Britain and Northern Ireland in 1927
. The Prime Minister of the United Kingdom is the head of the Government of the United Kingdom , and chairs Cabinet meetings . There is no specific date when
the office of Prime Minister first appeared , as the role was not created but rather evolved over a period of time . The term was used in the House of Commons
in 1805 and it was certainly in parliamentary use by the 1880s , and in 1905 the post of Prime Minister was officially given recognition in the order of
precedence .
</item 7>

<item 8>
List of Prime Ministers of the United Kingdom
Committee Scotland Government Parliament Wales Government National Assembly Reserved matters Northern Ireland Executive Assembly Administrative geography ( show
) England Subdivisions of England Local government in England Scotland Subdivisions of Scotland Local government in Scotland Wales Local government in Wales
Northern Ireland Local government in Northern Ireland Elections ( show ) Parliament constituencies Political parties Last election Next election Referendums
Foreign relations ( show ) History Foreign policy UK and the United Nations Politics of the European Union Commonwealth of Nations United Kingdom portal Other
countries Atlas Downing Street Historical rankings of Prime Ministers of the United Kingdom List of Prime Ministers of the United Kingdom by tenure Living Prime
Ministers of the United Kingdom Records of Prime Ministers of the United Kingdom List of current heads of government in the United Kingdom and dependencies
Royal prerogative in the United Kingdom List of British governments List of votes of no confidence in British governments List of United Kingdom general
elections British Prime Ministers -- Wikipedia book Premierships ^ Premierships of Benjamin Disraeli ^ Premierships of William Ewart Gladstone ^ Premiership of
Margaret Thatcher ^ Premiership of Tony Blair ^ Premiership of Gordon Brown ^ Premiership of David Cameron ^ Premiership of Theresa May Notes ( edit ) Coalition
government / hung parliament .
</item 8>

<item 9>
List of Prime Ministers of the United Kingdom
After the succession of King George I in 1714 , the arrangement of a commission of Lords of the Treasury as opposed to a single Lord High Treasurer became
permanent . For the next three years , the government was headed by Lord Townshend , who was appointed Secretary of State for the Northern Department .
Subsequently , Lord Stanhope and Lord Sunderland ran the government jointly , with Stanhope managing foreign affairs and Sunderland managing domestic affairs .
Stanhope died in February 1721 and Sunderland resigned two months later ; Townshend and Robert Walpole were then invited to form the next government . From that
point , the holder of the office of First Lord also unofficially held the status of Prime Minister . It was not until the Edwardian era that the title Prime
Minister was constitutionally recognised . With only few exceptions , the Prime Minister still holds the office of First Lord by constitutional convention .
Since 1721 ( edit ) By century and monarch 18th to 19th century George I George II George III George IV William IV Victoria 20th to 21st century Edward VII
George V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative (
</item 9>

<item 10>
List of Prime Ministers of the United Kingdom
`` Thatcher longest serving British prime minister '' . United Press International . Retrieved 26 June 2017 . Jump up ^ Burt ( 1874 ) , p. 106 . Jump up ^
Seaman ( 2003 ) . Jump up ^ chpt. 4 , Royal and Parliamentary Titles Act 1927 , 12 April 1927 . Retrieved on 3 September 2017 . ^ Jump up to : Leonard ( 2010 )
, p. 1 . ^ Jump up to : Carpenter ( 1992 ) , p. 37 . Jump up ^ Leonard ( 2010 ) , p. 47 . ^ Jump up to : Leonard ( 2010 ) , p. 65 . Jump up ^ `` Prime ministers
of the United Kingdom ( 1730 -- 2011 ) '' . Oxford Dictionary of National Biography ( online ed . ) . Oxford University Press . ( Subscription or UK public
library membership required . ) ^ Jump up to : Chisholm , Hugh , ed. ( 1911 ) . `` Treasury '' .
</item 10>
=== END ITEM id=a181-11494f8a6120a450 ===

=== ITEM id=a193-e0b5eae77f49be09 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him . `` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied about Robin Hood 's soul having been destroyed . Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental . Belle lives on Hook 's ship ; in a deal with Gold , the Evil Queen will not harm her or the baby . Emma tells Archie about her visions and notes that Regina is n't present in them and could be the hooded figure . 114 `` The Other Shoe '' Steve Pearlman Jane Espenson & Jerome Schwartz October 9 , 2016 ( 2016 - 10 - 09 ) 4.11 In the past in the Enchanted Forest , Cinderella meets Prince Thomas at the ball , but flees when she believes he loves her stepsister Clorinda ; Clorinda actually plans to elope with Thomas ' footman , Jacob , but her mother , Lady Tremaine , abducts her to the Land of Untold Stories after forcing Cinderella to betray Clorinda 's location . In Storybrooke , Ashley hopes to reconcile with Clorinda . Aided by the Evil Queen , Tremaine lures Ashley into a trap for her and Jacob . Ashley protects Jacob ; Clorinda reunites with him and forgives Cinderella , while Tremaine is arrested . Emma struggles to control her magic . Whale helps Jekyll recreate his serum , while the Evil Queen and Hyde form an alliance and she releases him . Snow White wants normality , but David secretly investigates his father 's death ; Gold provides information in exchange for David delivering a message to Belle . 115 `` Strange Case '' Alrick Riley David H. Goodman & Nelson Soler October 16 , 2016 ( 2016 - 10 - 16 ) 3.53 In the past in Victorian England , Jekyll petitions the Academy of Science for membership but Dr. Lydgate rejects him . Rumplestiltskin arrives and magically perfects Jekyll 's serum ; Jekyll transforms into Hyde , who blackmails Lydgate for Jekyll 's membership . Mary is Lydgate 's daughter and Jekyll 's unrequited love ; she and Hyde fall in love . Jekyll accidentally fatally defenestrates Mary ; he frames Hyde . In Storybrooke , Gold imprisons Belle on the ship to protect her from Hyde ; but it is Jekyll who tries to kill Belle in revenge for Rumplestiltskin 's role in Mary 's death . Hook saves Belle by killing Jekyll , and Hyde dies of identical wounds ; Regina asks Emma to kill her if necessary to stop the Evil Queen . Gold warns Belle that their child will need his protection . Mary Margaret resumes teaching , and her assistant Shirin inspires her to draw on her life as Snow . Shirin is secretly Princess Jasmine , working with the Oracle to search for Aladdin . 116 5 `` Street Rats '' Norman Buckley Edward Kitsis & Adam Horowitz October 23 , 2016 ( 2016 - 10 - 23 ) 3.40 In the past in Agrabah , Aladdin seeks a magical weapon for Jasmine , to save the kingdom from Jafar ; he is the weapon and a Savior . Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death . Aladdin rejects Jafar and saves Jasmine . In Storybrooke , the Oracle is murdered . Emma helps Jasmine , hoping Aladdin 's survival ensures her own . Having faked his death , Aladdin reveals himself to Emma and confesses that he doomed Agrabah by using the shears , which he gives to her ; she asks Hook to dispose of them , but he secretly keeps them . The Evil Queen impersonates Archie to learn of Emma 's visions , and convinces Zelena to turn Archie back into a cricket . 117 6 `` Dark Waters '' Robert Duncan Andrew Chambliss & Brigitte Hales October 30 , 2016 ( 2016 - 10 - 30 ) 3.06 During the Dark Curse , Captain Nemo abducts Hook and invites him to join the crew `` family '' of his submarine , Nautilus , in their quest for the `` Mysterious Island '' , which is the Land of Untold Stories . Hook discovers the first mate is his half - brother , Liam , and decides to leave to avoid confrontation ; but Liam attacks him , instead mortally wounding Nemo , who steps in . In the present , Archie is freed . Henry confronts Hook over lying about the shears , and Liam imprisons them on Nautilus . Hook opens up and Henry accepts him as family ; they save each other and dispose of the shears . Modern medicine saves Nemo ; he , Liam , and Hook make amends . Emma and Aladdin discuss being a Savior . Jasmine reveals that Agrabah vanished . Snow supports Belle at her ultrasound . The Evil Queen seduces Gold and offers the retrieved shears in exchange for help taking Snow 's heart . 118 7 `` Heartless '' Ralph Hemecker Jane Espenson November 6 , 2016 ( 2016 - 11 - 06 ) 3.56 In the past , fugitive Snow and shepherd David separately travel to Longbourn ; she to flee the kingdom and he to sell his struggling farm . Without ever seeing each other 's faces , David saves Snow from a bounty hunter and she repays him with money to save the farm , deciding not to flee . Their hands touch and a magical sapling is created . In the present , the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water unless Mary Margaret and David hand over their hearts . They learn the sapling could stop the Queen , who destroys it just after they find it . But it has shown them their previous meeting , and they conclude that their love can endure anything ; they surrender themselves . The Queen casts a sleeping curse on their hearts ; because they share Snow 's heart , the curse affects them both ; each can wake the other but immediately succumbs to the curse , so they can never be awake together . Regina uses Gold 's relationship with the Queen to separate the latter from Zelena , who earns Gold 's enmity by exposing his schemes to Belle . 119 8 `` I 'll Be Your Mirror '' Jennifer Lynch Jerome Schwartz & Leah Fong November 13 , 2016 ( 2016 - 11 - 13 ) 3.40 Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror , but she traps them there , where the Dragon is already imprisoned . Henry prepares for a date with Violet ; the Queen impersonates Regina and tries to manipulate him but he recognizes her . Emma and Regina discover Sidney was building a mirror portal out of the world and try and finish it , but the Evil Queen forces the Dragon to attack Emma and Regina , using his heart , and tries to make Henry kill the Dragon with the Hammer of Hephaestus ; he instead uses it to free his mothers . Mary Margaret and David adjust to living in shifts . Belle asks Zelena to help her escape Storybrooke via magical portal , and they form an alliance with Aladdin and Jasmine . Aladdin steals the Genie 's lamp and the Sorcerer 's wand from Gold 's shop , but Gold takes the wand back from Zelena and puts a magical tracking device on Belle . He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H. Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina continue to feud . Emma obtains the sword from her vision . 121 10 `` Wish You Were Here '' Ron Underwood Edward Kitsis & Adam Horowitz December 4 , 2016 ( 2016 - 12 - 04 ) 3.27 The Black Fairy abducts Gideon . Emma discovers the sword can kill the Evil Queen without hurting Regina . The Queen steals the lamp and wishes Emma to an alternate reality where she was never the Savior ; Regina follows her . David and his allies take the lamp from the Queen , who is turned into a serpent by the hooded figure , revealed to be the adult Gideon . Jasmine wishes herself and Aladdin to Agrabah . In the `` wish realm '' , Emma lives as a princess until Regina traumatically restores her memory . Regina frees Rumplestiltskin in exchange for a magic bean , but the portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood . 122 11 `` Tougher Than the Rest '' Billy Gierhart Edward Kitsis & Adam Horowitz March 5 , 2017 ( 2017 - 03 - 05 ) 3.03 In 1990 Minnesota , a young Emma is told by a teenager that she has the power to change her fate . In present day , after they miss their way out of the Wish realm , Emma and Regina find another option for escape when they come across August , who agrees to help create a new portal . Regina goes to find Robin Hood to see if he would 've been better off without her . After Regina finds Robin , they 're caught by the Sheriff of Nottingham and later by the Wish realm Rumple . Robin believes Regina when she tells him about their being in an alternate reality . Emma discovers that August was the teenager she met back in Minnesota and inspires him to complete the portal . Emma , Regina , and Robin return to Storybrooke , where Gideon explains to Gold and Belle his intention to kill Emma so that he can acquire her savior powers and kill the Black Fairy , a move that has David furious and Belle concerned . Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12 ) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried about whether David will give him his blessing to marry Emma , but David does . Meanwhile , Regina tries to get the Wish Realm Robin assimilated into Storybrooke , but soon questions why he agreed to follow her , as it turns out that he had something else planned after he steals a chest from the vault . The truth as to who really murdered Robert is revealed when August gives Hook missing pages he took from Henry 's book . When Hook sees an illustration of David 's dad , he recognizes him and recalls murdering him during a robbery in the Enchanted Forest so that there was no witness to his crime . 124 13 `` Ill - Boding Patterns '' Ron Underwood Andrew Chambliss & Dana Horgan March 19 , 2017 ( 2017 - 03 - 19 ) 2.71 During the First Ogres War , Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old . After every last soldier had been slain , Beowulf surrenders to death just as Rumplestiltskin intervenes , killing all the ogres . Hailed as a hero in the village , Beowulf is irked by Rumple 's new title . He creates a ruse to lead Rumple to his demise . Fearing he would not be able to follow through on protecting the villagers without Dark Magic , Rumple bestows his dagger upon Baelfire as an insurance policy . Following a turbulent exchange and threat to frame Rumplestiltskin as a murderer , Beowulf retreats . Feeling the pressure of his threat , Bae commands his father to kill him . Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence . In Storybrooke , Rumple recognizes a familiar pattern in Gideon , who makes strides in his quest to kill Emma Swan at the cost of The Blue Fairy 's vitality . Hook worries about the repercussions of telling Emma his newly unveiled secret . Meanwhile , Emma unknowingly forces a proposal after happening upon Hook 's engagement ring . Robin of Locksley attempts to flee the town , but to no avail . The Evil Queen escapes her reptilian confinement and indoctrinates Robin . 125 14 `` Page 23 '' Kate Woods David H. Goodman & Brigitte Hales March 26 , 2017 ( 2017 - 03 - 26 ) 2.85 In the Enchanted Forest , The Evil Queen is in search of a spell that will help her end Snow White for good , but her father has love in mind for his daughter and leads her on a quest to Cupid 's Arrow . However , The Evil Queen rebrands the arrow to follow hatred and discovers that the person she despises the most is herself . In Storybrooke , The Evil Queen and Regina face off in a final confrontation . Just when Regina wins the upper hand , she remembers the Evil Queen 's self - loating and evenly splits the light and the darkness in their hearts , to share their love and hate and allow the Evil Queen to reform . After Emma calls off the engagement , Hook decides to board the Nautilus with Captain Nemo . Hook later changes his mind following a conversation with Snow but , before he can return , Gideon hijacks the vessel and sends it out of Storybrooke . 126 15 `` A Wondrous Place '' Steve Pearlman Jane Espenson & Jerome Schwartz April 2 , 2017 ( 2017 - 04 - 02 ) 2.80 At a loss for hope , Regina suggests a `` girls ' night out '' for Emma with her and Snow . While at the bar , she confides in the mixologist and sheds a tear . Meanwhile , the Nautilus is transported to the Enchanted Forest , where Aladdin and Jasmine board . They have yet to locate Agrabah and are growing weary . After Hook and Jasmine confide in one another , the vessel is irrevocably damaged and Jasmine uses a genie wish to relocate the group to the island where Jafar is located . They discover a cabana belonging to Ariel and Prince Eric . Ariel has collected a genie lamp that houses Jafar . Jasmine is able to sway him into divulging the location of the hidden Agrabah before defeating him with magic dust . Through true love 's kiss , Aladdin and Jasmine , along with Hook and Ariel , are transported to the restored city and Aladdin 's genie curse is broken . Ariel supplies Hook with a means to reach Emma . Emma receives the message and Gideon , no longer masquerading as the mixologist , threatens that he will use her tears to keep Hook from returning to Storybrooke unless she helps him kill the Black Fairy . Flashbacks to Ariel and Jasmine 's first meeting show the princess falter under Jafar 's trickery . 127 16 `` Mother 's Little Helper '' Billy Gierhart Story by : Edward Kitsis & Adam Horowitz Teleplay by : Paul Karp April 9 , 2017 ( 2017 - 04 - 09 ) 2.60 In Storybrooke , Gold reveals to Emma and Snow that the Black Fairy is his mother and that the Dark Curse Regina cast was created by the Black Fairy . Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is controlling him to kill Emma . 128 17 `` Awake '' Sharat Raju Andrew Chambliss & Leah Fong April 16 , 2017 ( 2017 - 04 - 16 ) 2.51 Regina works feverishly to wake Snow and Charming from the Sleeping Curse . Gold comes face to face with the Black Fairy , the powerful enemy Emma must fight in the Final Battle . Hook , still in Neverland , meets up with Tiger Lily , who gives him a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy . Gold confronts the Black Fairy over her manipulating Gideon with his heart and she maintains that Storybrooke would be destroyed if the two of them fought . A pixie flower allows Emma to open a doorway to Neverland and rescue Hook , who then re-proposes properly . The townsfolk gather around Emma 's sleeping parents and each take small amounts of the Sleeping Curse unto themselves , diluting it and allowing Snow and David to finally reunite . In 1993 , a pixie flower awakens Snow and Charming from their cursed selves , but an awakened Rumplestiltskin tells them they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior . 129 18 `` Where Bluebirds Fly '' Michael Schultz David H. Goodman & Brigitte Hales April 23 , 2017 ( 2017 - 04 - 23 ) 2.69 In Zelena 's past , her childhood friend Stanum asks for her help in finding the Crimson Heart , a magical artifact that would be able to restore his human form after having been cursed by the Wicked Witch of the North to slowly turn into tin . Zelena 's refusal to give up her magic leads to her betraying Stanum , as she keeps the Heart and leaves him to become the Tin Man . In the Present , The Black Fairy tempts Zelena with an offer to join her , only to have Zelena oppose her instead . Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm . In the Present , as everyone looks for the missing wand , Gold brings Gideon and Emma into the Dream World to search for Gideon 's heart , where the events of the flashbacks unfold . Gold then confronts Fiona , but she reveals the real reason behind why he was abandoned . Gold later retrieves Gideon 's heart and tells everyone that the Black Fairy is dead , paving the way for Emma and Hook 's wedding . However , Gold and Fiona make a deal that will see the Final Battle go forward , with Emma being the Savior that will fight in it . Meanwhile , Regina helps Zelena adjust to life without her powers . 131 20 `` The Song in Your Heart '' Ron Underwood David H. Goodman & Andrew Chambliss May 7 , 2017 ( 2017 - 05 - 07 ) 2.87 In the past , Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength , which also spreads to Oz as well . When Regina becomes affected by the wish , she finds a way to end it before the first curse takes place , but the Blue Fairy , who granted Snow the wish , tells Snow and David that the wish will be part of Emma 's destiny . In Emma 's past , her chance to make use of her talent is held back by being told she 'll be alone . This would come back to taunt Emma in the present day , as Fiona returns to use that weakness to take Emma 's heart before unleashing a new curse on her wedding day , with help from Gold by freezing her family . When Henry discovers the page in the book that reveals the song in Emma 's heart will make her stronger , not only does it embolden Emma , it stops Fiona from crushing her heart , but Fiona still tells Emma that the battle will continue . On her wedding day , Emma and Hook finally marry , just in time for the dark clouds of the Black Fairy 's curse to erupt from the clock tower and engulf them at 6 : 00 PM , setting up the final battle . 132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle . Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their `` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity . Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
  Fact 0.244: Regina evenly splits the light and the darkness in their hearts, to share their love and hate.

Retrieved items:
<item 1>
Once Upon a Time (season 6)
the Bow '' `` Nimue '' `` Birth '' `` The Bear King '' `` Broken Heart '' `` Swan Song '' `` Souls of the Departed '' `` Labor of Love '' `` Devil 's Due '' ``
The Brothers Jones '' `` Our Decay '' `` Her Handsome Hero '' `` Ruby Slippers '' `` Sisters '' `` Firebird '' `` Last Rites '' `` Only You '' `` An Untold
Story '' Season 6 `` The Savior '' `` A Bitter Draught '' `` The Other Shoe '' `` Strange Case '' `` Street Rats '' `` Dark Waters '' `` Heartless '' `` I 'll
Be Your Mirror '' `` Changelings '' `` Wish You Were Here '' `` Tougher Than the Rest '' `` Murder Most Foul '' `` Ill - Boding Patterns '' `` Page 23 '' `` A
Wondrous Place '' `` Mother 's Little Helper '' `` Awake '' `` Where Bluebirds Fly '' `` The Black Fairy '' `` The Song in Your Heart '' `` The Final Battle ''
Season 7 `` Hyperion Heights '' `` A Pirate 's Life '' `` The Garden of Forking Paths '' `` Beauty '' `` Greenbacks '' `` Wake Up Call '' `` Eloise Gardener ''
`` Pretty
</item 1>

<item 2>
Once Upon a Time (season 6)
Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in
Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents
him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma
having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their
`` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity .
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
</item 2>

<item 3>
Once Upon a Time (season 6)
Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with
the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a
game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run
from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help
from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the
savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is
controlling him to kill Emma .
</item 3>

<item 4>
Once Upon a Time (season 6)
Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking
Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides
to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma
then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick
Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona
learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it
results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm .
</item 4>

<item 5>
Once Upon a Time (season 6)
overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz
September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In
Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of
a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him .
`` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied
about Robin Hood 's soul having been destroyed .
</item 5>

<item 6>
Once Upon a Time (season 6)
Once Upon a Time ( season 6 ) - wikipedia Once Upon a Time ( season 6 ) Jump to : navigation , search Once Upon a Time ( season 6 ) DVD cover Starring Ginnifer
Goodwin Jennifer Morrison Lana Parrilla Josh Dallas Emilie de Ravin Colin O'Donoghue Jared S. Gilmore Rebecca Mader Robert Carlyle Country of origin United
States No. of episodes 22 Release Original network ABC Original release September 25 , 2016 ( 2016 - 09 - 25 ) -- May 14 , 2017 ( 2017 - 05 - 14 ) Season
chronology ← Previous Season 5 Next → Season 7 List of Once Upon a Time episodes The sixth season of the American ABC fantasy - drama Once Upon a Time was
ordered on March 3 , 2016 . It debuted on September 25 , 2016 , and concluded on May 14 , 2017 . In January 2017 , it was stated that the sixth season would end
the main storyline , and for a seventh season , the series would be softly rebooted with a new storyline .
</item 6>

<item 7>
Once Upon a Time (season 6)
Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Regular 2.2 Recurring 2.3 Guest 3 Episodes 4 Production 4.1 Development 4.1. 1 Musical episode 4.2 Casting
5 Ratings 6 References 7 External links Premise ( edit ) The residents of Storybrooke are faced with threats from the Evil Queen and Mr. Hyde , both with
different agendas following the arrival of refugees from the Land of Untold Stories . This leads to events to test Emma 's savior abilities that results in the
arrival of the Black Fairy to begin the Final Battle that was prophesied centuries ago . As dark and light collides , a new , and darker curse is unleashed ,
leading to the fall of all the realms with magic . The Final Battle is fought and won , leading to the heroes earning their happy beginnings , but a new
adventure begins in the far future for an adult Henry involving his daughter Lucy , and another curse looming across the New Enchanted Forest inhabitants .
</item 7>

<item 8>
Once Upon a Time (season 6)
132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters
Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle
begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are
destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at
Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona
convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle .
</item 8>

<item 9>
Once Upon a Time (season 6)
-- Executive producers Kitsis and Horowitz on Once 's interpretation of Aladdin . Emilie de Ravin told fans that she will be returning for the sixth season . It
was also announced later that Sam Witwer and Hank Harris have signed a contract to play two new recurring characters on the sixth season of the show . On March
29 , 2016 , Lana Parrilla confirmed she would be returning for the sixth season . Robert Carlyle was confirmed to be returning for the sixth season as
Rumplestiltskin along with Rebecca Mader as Zelena and Jared Gilmore as Henry Mills . It was announced that Giles Matthey was cast as Morpheus , who is slated
to appear in the first episode of the season . On July 20 , it was announced that Craig Horner would be portraying the Count of Monte Cristo , who was
introduced in the second episode of the season .
</item 9>

<item 10>
Once Upon a Time (season 6)
Existing fictional characters introduced to the series during the season include Aladdin , Princess Jasmine , the Count of Monte Cristo , Captain Nemo , Lady
Tremaine , Beowulf , Tiger Lily and the Tin Man . Original new characters include Gideon , the Black Fairy , Mary Lydgate , and Robert . The show also
reintroduced Jafar and Dr. Arthur Lydgate , who previously appeared in Once Upon a Time in Wonderland . This season also marks the final appearance of Emma Swan
( Jennifer Morrison ) as a series regular . Morrison announced she would be departing the series after the sixth - season finale , but if the series receives a
seventh season renewal she has agreed to appear in at least one episode . After serving as a series regular for two seasons , Rebecca Mader also announced that
season six would be her last on the show as a regular . Shortly after , Adam Horowitz and Edward Kitsis announced that original cast members Ginnifer Goodwin ,
Josh Dallas , and Jared Gilmore , as well as Emilie de Ravin who joined the main cast in season two , would also exit the show at the end of the season .
</item 10>
=== END ITEM id=a193-e0b5eae77f49be09 ===

=== ITEM id=a143-50c9bd1e1cc2da58 ===
Gold sentences and facts:
Sentence 0: No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 ) Labour 27 April 1904 18 August 1904 Bland , NSW , 1901 -- 1906 South Sydney , NSW , 1906 -- 1910 ( retired ) -- Watson George Reid ( 1845 -- 1918 ) Free Trade 18 August 1904 5 July 1905 East Sydney , NSW , 1901 -- 1909 ( resigned ) -- Reid ( 2 ) Alfred Deakin ( 1856 -- 1919 ) Protectionist 5 July 1905 13 November 1908 Ballaarat , Vic , 1901 -- 1913 ( retired ) -- 2nd Deakin 1906 3rd Deakin 5 Andrew Fisher ( 1862 -- 1928 ) Labour 13 November 1908 2 June 1909 Wide Bay , Qld , 1901 -- 1915 ( resigned ) -- 1st Fisher ( 2 ) Alfred Deakin ( 1856 -- 1919 ) Commonwealth Liberal 2 June 1909 29 April 1910 Ballaarat , Vic , 1901 -- 1913 ( retired ) -- 4th Deakin ( 5 ) Andrew Fisher ( 1862 -- 1928 ) Labor 29 April 1910 24 June 1913 Wide Bay , Qld , 1901 -- 1915 ( resigned ) 1910 2nd Fisher 6 Joseph Cook ( 1860 -- 1947 ) Commonwealth Liberal 24 June 1913 17 September 1914 Parramatta , NSW , 1901 -- 1921 ( resigned ) 1913 Cook ( 5 ) Andrew Fisher ( 1862 -- 1928 ) Labor 17 September 1914 27 October 1915 Wide Bay , Qld , 1901 -- 1915 ( resigned ) 1914 3rd Fisher Billy Hughes ( 1862 -- 1952 ) Labor 27 October 1915 14 November 1916 West Sydney , NSW , 1901 -- 1917 Bendigo , Vic , 1917 -- 1922 North Sydney , NSW , 1922 -- 1949 Bradfield , NSW , 1949 -- 1952 ( died ) -- 1st Hughes 7 National Labor 14 November 1916 17 February 1917 -- 2nd Hughes Nationalist 17 February 1917 9 February 1923 -- 3rd Hughes 1917 4th Hughes 1919 5th Hughes 8 Stanley Bruce ( 1883 -- 1967 ) Nationalist ( Coalition ) 9 February 1923 22 October 1929 Flinders , Vic , 1918 -- 1929 ( defeated ) ; 1931 -- 1933 ( resigned ) 1922 1st Bruce 1925 2nd Bruce 1928 3rd Bruce 9 James Scullin ( 1876 -- 1953 ) Labor 22 October 1929 6 January 1932 Corangamite , Vic , 1910 -- 1913 ( defeated ) Yarra , Vic , 1922 -- 1949 ( retired ) 1929 Scullin 10 Joseph Lyons ( 1879 -- 1939 ) United Australia ( Coalition ) 6 January 1932 7 April 1939 † Wilmot , Tas , 1929 -- 1939 ( died ) 1931 1st Lyons 1934 2nd Lyons -- 3rd Lyons 1937 4th Lyons 11 Sir Earle Page ( 1880 -- 1961 ) Country ( Coalition ) 7 April 1939 26 April 1939 Cowper , NSW 1919 -- 1961 ( defeated ) -- Page 12 Robert Menzies ( 1894 -- 1978 ) United Australia ( Coalition ) 26 April 1939 28 August 1941 Kooyong , Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October 1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 † Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945 Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW , 1928 -- 1931 ( defeated ) ; 1940 -- 1951 ( died ) -- 1st Chifley 1946 2nd Chifley ( 12 ) Sir Robert Menzies ( 1894 -- 1978 ) Liberal ( Coalition ) 19 December 1949 26 January 1966 Kooyong , Vic , 1934 -- 1966 ( resigned ) 1949 4th Menzies 1951 5th Menzies 1954 6th Menzies 1955 7th Menzies 1958 8th Menzies 1961 9th Menzies 1963 10th Menzies 17 Harold Holt ( 1908 -- 1967 ) Liberal ( Coalition ) 26 January 1966 19 December 1967 † Fawkner , Vic , 1935 -- 1949 Higgins , Vic , 1949 -- 1967 ( disappeared ) -- 1st Holt 1966 2nd Holt 18 John McEwen ( 1900 -- 1980 ) Country ( Coalition ) 19 December 1967 10 January 1968 Echuca , Vic , 1934 -- 1937 Indi , Vic , 1937 -- 1949 Murray , Vic , 1949 -- 1971 ( resigned ) -- McEwen 19 John Gorton ( 1911 -- 2002 ) Liberal ( Coalition ) 10 January 1968 10 March 1971 Senator 1950 -- 1968 ( resigned ) MP for Higgins , Vic , 1968 -- 1975 ( retired ) -- 1st Gorton 1969 2nd Gorton 20 William McMahon ( 1908 -- 1988 ) Liberal ( Coalition ) 10 March 1971 5 December 1972 Lowe , NSW , 1949 -- 1982 ( resigned ) -- McMahon 21 Gough Whitlam ( 1916 -- 2014 ) Labor 5 December 1972 11 November Werriwa , NSW , 1952 -- 1978 ( resigned ) 1972 1st Whitlam -- 2nd Whitlam 3rd Whitlam 22 Malcolm Fraser ( 1930 -- 2015 ) Liberal ( Coalition ) 11 November 11 March Wannon , Vic , 1955 -- 1983 ( resigned ) -- 1st Fraser 2nd Fraser 1977 3rd Fraser 1980 4th Fraser 23 Bob Hawke ( 1929 -- ) Labor 11 March 20 December 1991 Wills , Vic , 1980 -- 1992 ( resigned ) 1st Hawke 1984 2nd Hawke 3rd Hawke 1990 4th Hawke 24 Paul Keating ( 1944 -- ) Labor 20 December 1991 11 March Blaxland , NSW , 1969 -- 1996 ( resigned ) -- 1st Keating 1993 2nd Keating 25 John Howard ( 1939 -- ) Liberal ( Coalition ) 11 March 3 December 2007 Bennelong , NSW , 1974 -- 2007 ( defeated ) 1st Howard 1998 2nd Howard 2001 3rd Howard 4th Howard 26 Kevin Rudd ( 1957 -- ) Labor 3 December 2007 24 June Griffith , Qld , 1998 -- 2013 ( resigned ) 2007 1st Rudd 27 Julia Gillard ( 1961 -- ) Labor 24 June 27 June 2013 Lalor , Vic , 1998 -- 2013 ( retired ) -- 1st Gillard 2nd Gillard ( 26 ) Kevin Rudd ( 1957 -- ) Labor 27 June 2013 18 September 2013 Griffith , Qld , 1998 -- 2013 ( resigned ) -- 2nd Rudd 28 Tony Abbott ( 1957 -- ) Liberal ( Coalition ) 18 September 2013 15 September 2015 Warringah , NSW , since 1994 2013 Abbott 29 Malcolm Turnbull ( 1954 -- ) Liberal ( Coalition ) 15 September 2015 Incumbent Wentworth , NSW , since 2004 -- 1st Turnbull 2016 2nd Turnbull
  Fact 0.46: Billy Hughes represented West Sydney, NSW, from 1901 to 1917.

Retrieved items:
<item 1>
Prime Minister of Australia
Fraser accepted these terms and immediately advised a double dissolution . An election was called for 13 December , which the Liberal Party won in its own right
( although the Liberals governed in a coalition with the Country Party ) . Powers and role ( edit ) The first Prime Minister of Australia , Edmund Barton (
sitting second from left ) , with his Cabinet , 1901 . Most of the Prime Minister 's powers derive from being head of Government . In practice , the Federal
Executive Council will act to ratify all decisions made by the cabinet and decisions of the cabinet will always require the support of the Prime Minister . The
powers of the governor - general to grant Royal Assent to legislation , to dissolve and prorogue parliament , to call elections and to make appointments are
exercised on the advice of the Prime Minister .
</item 1>

<item 2>
Prime Minister of Australia
The leaders of the three parties , Alfred Deakin , George Reid and Chris Watson each served as Prime Minister before losing a vote of confidence . As a result
of the Labor Party 's split over conscription , Billy Hughes and his supporters were expelled from the Labor Party in November 1916 . He subsequently continued
on as prime minister at the head of the new National Labor Party , which had only 14 members out of a total of 75 in the House of Representatives . The
Commonwealth Liberal Party -- despite still forming the Official Opposition -- provided confidence and supply until February 1917 , when the two parties agree
to merge and form the Nationalist Party . During the 1975 constitutional crisis , on 11 November 1975 , the governor - general , Sir John Kerr , dismissed the
Labor Party 's Gough Whitlam as Prime Minister . Despite Labor holding a majority in the House of Representatives , Kerr appointed the Leader of the Opposition
, Liberal leader Malcolm Fraser as caretaker Prime Minister , conditional on the passage of the Whitlam government 's Supply bills through the Senate and the
calling of an election for both houses of parliament . Fraser accepted these terms and immediately advised a double dissolution .
</item 2>

<item 3>
Prime Minister of Australia
The governor - general , Lord Casey , commissioned the Leader of the Country Party , John McEwen , to form a government until the Liberal Party elected a new
leader . McEwen was Prime Minister for 23 days , until the election of ( then Senator ) John Gorton . Excluding the brief transition periods during changes of
government or leadership elections , there have only been a handful of cases where someone other than the leader of the majority party in the House of
Representatives was Prime Minister : Federation occurred on 1 January 1901 , but elections for the first parliament were not scheduled until late March . In the
interim , an unelected caretaker government was necessary . In what is now known as the Hopetoun Blunder , the governor - general , Lord Hopetoun , invited Sir
William Lyne , the premier of the most populous state , New South Wales , to form a government . Lyne was unable to do so and returned his commission in favour
of Edmund Barton , who became the first Prime Minister and led the inaugural government into and beyond the election . During the second parliament , three
parties ( Free Trade , Protectionist and Labor ) had roughly equal representation in the House of Representatives .
</item 3>

<item 4>
Prime Minister of Australia
Other commentators argue that the Governor General acted improperly in 1975 as Whitlam still retained the confidence of the House of Representatives , and there
are no generally accepted conventions to guide the use of the Governor General 's reserve powers in this circumstance. . However , there is no constitutional
requirement that the Prime Minister sit in the House of Representatives , or even be a member of the federal parliament ( subject to a constitutionally
prescribed limit of three months ) , though by convention this is always the case . The only case where a member of the Senate was appointed Prime Minister was
John Gorton , who subsequently resigned his Senate position and was elected as a member of the House of Representatives ( Senator George Pearce was acting Prime
Minister for seven months in 1916 while Billy Hughes was overseas ; and Senator Mathias Cormann acted for a few days in February 2018 while Malcolm Turnbull was
overseas ) . Malcolm Turnbull has held the office of Prime Minister since 15 September 2015 . He received his commission after replacing Tony Abbott as the
leader of the Liberal Party , the dominant party in the Coalition government , following the outcome of the September 2015 Liberal leadership ballot .
</item 4>

<item 5>
Prime Minister of Australia
Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October
1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 †
Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945
Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW ,
</item 5>

<item 6>
Prime Minister of Australia
NSW , 1928 -- 1931 ( defeated ) ; 1940 -- 1951 ( died ) -- 1st Chifley 1946 2nd Chifley ( 12 ) Sir Robert Menzies ( 1894 -- 1978 ) Liberal ( Coalition ) 19
December 1949 26 January 1966 Kooyong , Vic , 1934 -- 1966 ( resigned ) 1949 4th Menzies 1951 5th Menzies 1954 6th Menzies 1955 7th Menzies 1958 8th Menzies
1961 9th Menzies 1963 10th Menzies 17 Harold Holt ( 1908 -- 1967 ) Liberal ( Coalition ) 26 January 1966 19 December 1967 † Fawkner , Vic , 1935 -- 1949 Higgins
, Vic ,
</item 6>

<item 7>
Prime Minister of Australia
and Administrators Premiers and Chief Ministers Parliaments and Assemblies State courts State electoral systems Local government ( show ) NSW Vic Qld WA SA Tas
NT Related topics ( show ) Regions Foreign relations Defence Force Public holidays Republicanism Other countries Atlas Australia 's first Prime Minister ,
Edmund Barton at the central table in the House of Representatives in 1901 . The Prime Minister of Australia is appointed by the Governor - General of Australia
under Section 64 of the Australian Constitution , which empowers the Governor - General , as the official representative of the monarch , to appoint government
ministers of state and requires them to be members of the House of Representatives or the Senate , or become members within three months of the appointment .
The Prime Minister and Treasurer are traditionally members of the House , but the Constitution does not have such a requirement . Before being sworn in as a
Minister of the Crown , a person must first be sworn in as a member of the Federal Executive Council if they are not already a member . Membership of the
Federal Executive Council entitles the member to the style of The Honourable ( usually abbreviated to The Hon ) for life , barring exceptional circumstances .
The senior members of the Executive Council constitute the Cabinet of Australia .
</item 7>

<item 8>
Prime Minister of Australia
The formal title of the portfolio has always been simply `` Prime Minister '' , except for the period of the Fourth Deakin Ministry ( June 1909 to April 1910 )
, when it was known as `` Prime Minister ( without portfolio ) '' . If a government can not get its appropriation ( budget ) legislation passed by the House of
Representatives , or the House passes a vote of `` no confidence '' in the government , the Prime Minister is bound by convention to immediately advise the
Governor - General to dissolve the House of Representatives and hold a fresh election . Following a resignation in other circumstances , or the death of a Prime
Minister , the governor - general will generally appoint as Prime Minister the person elected as leader by the governing party or , in the case of a coalition ,
the senior party in the coalition . There have been four notable exceptions to this : When Joseph Lyons , Prime Minister and leader of the United Australia
Party ( UAP ) , died suddenly in April 1939 , the governor - general , Lord Gowrie , called on Sir Earle Page to become caretaker Prime Minister . Page was the
leader of the smaller party in the governing coalition , the Country Party .
</item 8>

<item 9>
Prime Minister of Australia
NSW , 1969 -- 1996 ( resigned ) -- 1st Keating 1993 2nd Keating 25 John Howard ( 1939 -- ) Liberal ( Coalition ) 11 March 3 December 2007 Bennelong , NSW , 1974
-- 2007 ( defeated ) 1st Howard 1998 2nd Howard 2001 3rd Howard 4th Howard 26 Kevin Rudd ( 1957 -- ) Labor 3 December 2007 24 June Griffith , Qld , 1998 -- 2013
( resigned ) 2007 1st Rudd 27 Julia Gillard ( 1961 -- ) Labor 24 June 27 June 2013 Lalor , Vic , 1998 -- 2013 ( retired ) -- 1st Gillard 2nd Gillard ( 26 )
Kevin Rudd ( 1957 -- ) Labor 27 June 2013 18 September 2013 Griffith , Qld , 1998 -- 2013 ( resigned ) -- 2nd Rudd 28 Tony Abbott ( 1957 -- ) Liberal (
Coalition ) 18 September 2013 15 September 2015 Warringah ,
</item 9>

<item 10>
Prime Minister of Australia
He served under the United Australia Party and the Liberal Party respectively . The shortest - serving Prime Minister was Frank Forde , who was appointed to the
position on 6 July 1945 after the death of John Curtin , and served until 13 July 1945 when Ben Chifley was elected leader of the Australian Labor Party .
Parties Australian Labor Party Liberal Party of Australia Australian Country Party Nationalist Party of Australia United Australia Party Commonwealth Liberal
Party National Labor Party Free Trade Party Protectionist Party No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won
Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred
Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 )
Labour 27 April 1904 18 August 1904 Bland , NSW ,
</item 10>
=== END ITEM id=a143-50c9bd1e1cc2da58 ===

=== ITEM id=a142-a5cf9b2d1c099b39 ===
Gold sentences and facts:
Sentence 0: St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok , Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes . Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
  Fact 1.14: Steubenville, Ohio is a location used for some mill and neighborhood shots

Retrieved items:
<item 1>
The Deer Hunter
The Deer Hunter - wikipedia The Deer Hunter Jump to : navigation , search For other uses , see Deer Hunter ( disambiguation ) . The Deer Hunter UK theatrical
release poster Directed by Michael Cimino Produced by Barry Spikings Michael Deeley Michael Cimino John Peverall Screenplay by Deric Washburn Story by Deric
Washburn Michael Cimino Louis Garfinkle Quinn K. Redeker Starring Robert De Niro John Cazale John Savage Meryl Streep Christopher Walken Music by Stanley Myers
Cinematography Vilmos Zsigmond Edited by Peter Zinner Production company EMI Distributed by Universal Pictures Release date December 8 , 1978 ( 1978 - 12 - 08 )
( Los Angeles ) February 23 , 1979 ( 1979 - 02 - 23 ) ( United States ) Running time 183 minutes Country United States Language English Russian Vietnamese
French Budget $15 million Box office $49 million The Deer Hunter is a 1978 American epic war drama film co-written and directed by Michael Cimino about a trio
of Russian American steelworkers whose lives are changed forever after they fight in the Vietnam War .
</item 1>

<item 2>
The Deer Hunter
Additionally , director Cimino instructed the props department to fashion complete Pennsylvania IDs for each of them , including driver 's licenses , medical
cards , and various other pieces of paraphernalia , in order to enhance each actor 's sense of his character . Pre-production ( edit ) There has been
considerable debate , controversy , and conflicting stories about how The Deer Hunter was initially developed and written . Director and co-writer Michael
Cimino , writer Deric Washburn , and producers Barry Spikings and Michael Deeley all have different versions of how the film came to be . Development ( edit )
In 1968 , the record company EMI formed a new company called EMI Films , headed by producers Barry Spikings and Michael Deeley . Deeley purchased the first
draft of a spec script called The Man Who Came to Play , written by Louis Garfinkle and Quinn K. Redeker , for $19,000 . The spec script was about people who go
to Las Vegas to play Russian roulette . `` The screenplay had struck me as brilliant , '' wrote Deeley , `` but it was n't complete .
</item 2>

<item 3>
The Deer Hunter
In the end , it took 51 minutes . Deeley believes that Cimino always planned to make this prologue last for an hour , and `` the plan was to be advanced by
stealth rather than straight dealing . '' At this point in the production , nearly halfway through principal photography , Cimino was already over budget , and
producer Spikings could tell from the script that shooting the extended scene could sink the project . The bar and the steel mill ( edit ) The bar was specially
constructed in an empty storefront in Mingo Junction , Ohio for $25,000 ; it later became an actual saloon for local steel mill workers . U.S. Steel allowed
filming inside its Cleveland mill , including placing the actors around the furnace floor , only after securing a $5 million insurance policy . Other filming
took place in Pittsburgh . Hunting the Deer ( edit ) The first deer to be shot was depicted in a `` gruesome close - up '' , although he was hit with a
tranquilizer dart . The stag that Michael allows to get away later was the same one used on TV commercials for the Connecticut Life Insurance Company .
</item 3>

<item 4>
The Deer Hunter
I 'd been working 20 hours a day for a month . I got on the plane the next day , and I went back to Manhattan and my carpenter job . '' Deeley 's reaction to
the revised script ( edit ) Deeley felt the revised script , now called The Deer Hunter , broke fresh ground for the project . The protagonist in the Redeker /
Garfinkle script , Merle , was an individual who sustained a bad injury in active service and was damaged psychologically by his violent experiences , but was
nevertheless a tough character with strong nerves and guts . Cimino and Washburn 's revised script distilled the three aspects of Merle 's personality and
separated them out into three distinct characters . They became three old friends who grew up in the same small industrial town and worked in the same steel
mill , and in due course were drafted together to Vietnam . In the original script , the roles of Merle ( later renamed Mike ) and Nick were reversed in the
last half of the film . Nick returns home to Linda , while Mike remains in Vietnam , sends money home to help Steven , and meets his tragic fate at the Russian
roulette table .
</item 4>

<item 5>
The Deer Hunter
His direction to his actors : `` You put the gun to your head , Chris , you shoot , you fall over and Bobby cradles your head . '' Filming locations ( edit )
Lemko Hall St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko
Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces
in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok ,
Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots .
</item 5>

<item 6>
The Deer Hunter
2 Washburn 's claim 3.2. 3 Deeley 's reaction to the revised script 4 Filming 4.1 The wedding scenes 4.2 The bar and the steel mill 4.3 Hunting the deer 4.4
Vietnam and the Russian roulette scenes 4.5 Filming locations 5 Post-production 5.1 Sound design 5.2 Previews 6 Soundtrack 6.1 Selected tracks 7 Release 8
Analysis 8.1 Controversy over Russian roulette 8.1. 1 Critics ' response 8.1.
</item 6>

<item 7>
The Deer Hunter
During the helicopter stunt , the runners caught on the rope bridge as the helicopter rose , it threatened to seriously injure De Niro and Savage . The actors
gestured and yelled furiously to the crew in the helicopter to warn them . Footage of this is included in the film . According to Cimino , De Niro requested a
live cartridge in the revolver for the scene in which he subjects John Cazale 's character to an impromptu game of Russian roulette , to heighten the intensity
of the situation . Cazale agreed without protest , but obsessively rechecked the gun before each take to make sure that the live round was n't next in the
chamber . While appearing later in the film , the first scenes shot upon arrival in Thailand were the hospital sequences between Walken and the military doctor
. Deeley believed that this scene was `` the spur that would earn him an Academy Award . '' In the final scene in the gambling den between Mike and Nick ,
Cimino had Walken and De Niro improvise in one take .
</item 7>

<item 8>
The Deer Hunter
The reception scene was filmed at nearby Lemko Hall . The amateur extras lined up for the crowded wedding - dance sequences drank real liquor and beer . The
scenes were filmed in the summer , but were set in the fall . To accomplish a look of fall , individual leaves were removed from deciduous trees . Zsigmond also
had to desaturate the colors of the exterior shots , partly in camera and in the laboratory processing . The production manager asked each of the Russian
immigrant extras to bring to the location a gift - wrapped box to double for wedding presents . The manager figured if the extras did this , not only would the
production save time and money , but the gifts would also look more authentic . Once the unit wrapped and the extras disappeared , the crew discovered to their
amusement that the boxes were n't empty but filled with real presents , from china to silverware . `` Who got to keep all these wonderful offerings , '' wrote
Deeley `` is a mystery I never quite fathomed . '' Cimino originally claimed that the wedding scene would take up 21 minutes of screen time . In the end , it
took 51 minutes .
</item 8>

<item 9>
The Deer Hunter
Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State
Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
Post-production ( edit ) By this point , The Deer Hunter had cost $13 million and the film still had to go through an arduous post-production . Film editor
Peter Zinner was given 600,000 feet of printed film to edit , a monumental task at the time . Producers Spikings and Deeley were pleased with the first cut ,
which ran for three and a half hours . `` We were thrilled by what we saw , '' wrote Deeley , `` and knew that within the three and a half hours we watched
there was a riveting film . '' Executives from Universal , including Lew Wasserman and Sid Sheinberg , were not very enthusiastic . `` I think they were shocked
, '' recalled Spikings . `` What really upset them was ' God Bless America ' .
</item 9>

<item 10>
The Deer Hunter
( Television Production ) . Chicago , IL : WTTW . ^ Jump up to : The Deer Hunter ( 1978 ) -- Filming locations Jump up ^ `` Lemko Hall '' . The Encyclopedia of
Cleveland History . May 24 , 2002 . Jump up ^ `` 7 Movie Locations in Bangkok - Bangkok.com Magazine '' . Bangkok.com . Retrieved 2016 - 10 - 17 . Jump up ^ Nam
, Suzanne ( 2012 ) . Moon Thailand . Avalon Travel . p. 320 . ISBN 9781598809695 . Jump up ^ Krause , Mariella ( 2013 ) . Lonely Planet Pacific Northwest 's
Best Trips . Lonely Planet . p. 118 . ISBN 1741798159 . Jump up ^ `` The Deer Hunter film locations ( 1978 ) '' . Movie-locations.com . Retrieved 2016 - 10 - 17
. Jump up ^ Randy Fox Writer and photographer ( 2012 - 05 - 05 ) .
</item 10>
=== END ITEM id=a142-a5cf9b2d1c099b39 ===

=== ITEM id=a185-97242bf8a40f049f ===
Gold sentences and facts:
Sentence 0: Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name ( Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741 7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743 -- First Lord of the Treasury Whig Carteret 7002502000000000000 ♠ 1 year and 137 days ​ The Right Honourable Henry Pelham FRS MP for Sussex ( 1694 -- 1754 ) 27 August 1743 6 March 1754 1747 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Broad Bottom I Broad Bottom II 7003384500000000000 ♠ 10 years and 192 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 16 March 1754 16 November 1756 1754 First Lord of the Treasury Leader of the House of Lords Whig Newcastle I 7002977000000000000 ♠ 2 years and 246 days His Grace William Cavendish 4th Duke of Devonshire KG PC ( 1720 -- 1764 ) 16 November 1756 25 June 1757 -- First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Devonshire ​ 1757 Caretaker 7002222000000000000 ♠ 222 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 2 July 1757 26 May 1762 1761 First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Newcastle George III ( 1760 -- 1820 ) 7003179000000000000 ♠ 4 years and 329 days The Right Honourable John Stuart 3rd Earl of Bute KG PC ( 1713 -- 1792 ) 26 May 1762 8 April 1763 -- First Lord of the Treasury Leader of the House of Lords Tory Bute ( Tory -- Whig ) 7002318000000000000 ♠ 318 days The Right Honourable George Grenville MP for Buckingham ( 1712 -- 1770 ) 16 April 1763 13 July 1765 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig ( Grenvillite ) Grenville ( Whig -- Tory ) 7002820000000000000 ♠ 2 years and 89 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 13 July 1765 30 July 1766 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham I 7002383000000000000 ♠ 1 year and 18 days The Right Honourable William Pitt 1st Earl of Chatham PC FRS MP for Bath ( 1708 -- 1778 ) 30 July 1766 14 October 1768 1768 Lord Keeper of the Privy Seal Whig ( Chathamite ) Chatham ( Whig -- Tory ) 7002808000000000000 ♠ 2 years and 77 days His Grace Augustus FitzRoy 3rd Duke of Grafton KG PC ( 1735 -- 1811 ) 14 October 1768 28 January 1770 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Grafton 7002472000000000000 ♠ 1 year and 107 days The Right Honourable Frederick North Lord North KG MP for Banbury ( 1732 -- 1792 ) 28 January 1770 22 March 1782 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Northite ) North 1774 1780 7003443700000000000 ♠ 12 years and 54 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 27 March 1782 1 July 1782 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham II 7001970000000000000 ♠ 97 days The Right Honourable William Petty 2nd Earl of Shelburne KG PC ( 1737 -- 1805 ) 4 July 1782 2 April 1783 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Shelburne ( Whig -- Tory ) 7002273000000000000 ♠ 273 days His Grace William Cavendish - Bentinck 3rd Duke of Portland PC FRS ( 1738 -- 1809 ) 2 April 1783 19 December 1783 -- First Lord of the Treasury Leader of the House of Lords Whig Fox -- North 7002262000000000000 ♠ 262 days The Right Honourable William Pitt the Younger MP for Appleby -- Cambridge University ( 1759 -- 1806 ) 19 December 1783 14 March 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt I 1784 1790 1796 7003629500000000000 ♠ 17 years and 86 days The Right Honourable Henry Addington MP for Devizes ( 1757 -- 1844 ) 17 March 1801 10 May 1804 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Addington 1802 7003115100000000000 ♠ 3 years and 55 days The Right Honourable William Pitt the Younger MP for Cambridge University ( 1759 -- 1806 ) 10 May 1804 23 January 1806 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt II 7002624000000000000 ♠ 1 year and 259 days The Right Honourable William Grenville 1st Baron Grenville PC PC ( Ire ) ( 1759 -- 1834 ) 11 February 1806 31 March 1807 First Lord of the Treasury Leader of the House of Lords Whig All the Talents ( Whig -- Tory ) 1806 7002414000000000000 ♠ 1 year and 49 days His Grace William Cavendish - Bentinck 3rd Duke of Portland KG PC FRS ( 1738 -- 1809 ) 31 March 1807 4 October 1809 First Lord of the Treasury Tory ( Pittite ) Portland II 1807 7002919000000000000 ♠ 2 years and 188 days The Right Honourable Spencer Perceval KC MP for Northampton ( 1762 -- 1812 ) 4 October 1809 11 May 1812 -- First Lord of the Treasury Chancellor of the Exchequer Chancellor of the Duchy of Lancaster Leader of the House of Commons Tory ( Pittite ) Perceval 7002951000000000000 ♠ 2 years and 221 days The Right Honourable Robert Jenkinson 2nd Earl of Liverpool KG PC FRS ( 1770 -- 1828 ) 8 June 1812 9 April 1827 1812 First Lord of the Treasury Leader of the House of Lords Tory ( Pittite ) Liverpool 1818 1820 George IV ( 1820 -- 1830 ) 1826 7003541900000000000 ♠ 14 years and 306 days The Right Honourable George Canning FRS MP for Seaford ( 1770 -- 1827 ) 10 April 1827 8 August 1827 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Canningite ) Canning ( Canningite -- Whig ) 7002121000000000000 ♠ 121 days The Right Honourable Frederick John Robinson 1st Viscount Goderich PC ( 1782 -- 1859 ) 31 August 1827 21 January 1828 -- First Lord of the Treasury Leader of the House of Lords Tory ( Canningite ) Goderich 7002144000000000000 ♠ 144 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 22 January 1828 16 November 1830 First Lord of the Treasury Leader of the House of Lords Tory Wellington -- Peel 1830 William IV ( 1830 -- 1837 ) 7003103000000000000 ♠ 2 years and 299 days The Right Honourable Charles Grey 2nd Earl Grey KG PC ( 1764 -- 1845 ) 22 November 1830 9 July 1834 First Lord of the Treasury Leader of the House of Lords Whig Grey 1831 1832 7003132600000000000 ♠ 3 years and 230 days The Right Honourable William Lamb 2nd Viscount Melbourne PC ( 1779 -- 1848 ) 16 July 1834 14 November 1834 -- First Lord of the Treasury Leader of the House of Lords Whig Melbourne I 7002122000000000000 ♠ 122 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 14 November 1834 10 December 1834 -- First Lord of the Treasury Sec. of State for the Home Department Sec. of State for Foreign Affairs Sec. of State for War and the Colonies Leader of the House of Lords Tory Wellington Caretaker 7001270000000000000 ♠ 27 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 10 December 1834 8 April 1835 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Conservative Peel I 7002120000000000000 ♠ 120 days The Right Honourable William Lamb 2nd Viscount Melbourne PC FRS ( 1779 -- 1848 ) 18 April 1835 30 August 1841 1835 First Lord of the Treasury Leader of the House of Lords Whig Melbourne II 1837 Victoria ( 1837 -- 1901 ) 7003232700000000000 ♠ 6 years and 135 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 30 August 1841 29 June 1846 1841 First Lord of the Treasury Leader of the House of Commons Conservative Peel II 7003176500000000000 ♠ 4 years and 304 days The Right Honourable Lord John Russell FRS MP for the City of London ( 1792 -- 1878 ) 30 June 1846 21 February 1852 1847 First Lord of the Treasury Leader of the House of Commons Whig Russell I 7003206300000000000 ♠ 5 years and 237 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 23 February 1852 17 December 1852 First Lord of the Treasury Leader of the House of Lords Conservative Who ? Who ? 1852 7002299000000000000 ♠ 299 days The Right Honourable George Hamilton - Gordon 4th Earl of Aberdeen KT FRSE FRS PC FSA Scot ( 1784 -- 1860 ) 19 December 1852 30 January 1855 -- First Lord of the Treasury Leader of the House of Lords Peelite Aberdeen ( Peelite -- Whig ) 7002773000000000000 ♠ 2 years and 43 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 6 February 1855 19 February 1858 First Lord of the Treasury Leader of the House of Commons Whig Palmerston I 1857 7003111000000000000 ♠ 3 years and 14 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 20 February 1858 11 June 1859 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli II 7002477000000000000 ♠ 1 year and 112 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 12 June 1859 18 October 1865 1859 First Lord of the Treasury Leader of the House of Commons Liberal Palmerston II 1865 7003232100000000000 ♠ 6 years and 129 days The Right Honourable John Russell 1st Earl Russell KG PC FRS ( 1792 -- 1878 ) 29 October 1865 26 June 1866 -- First Lord of the Treasury Leader of the House of Lords Liberal Russell II 7002241000000000000 ♠ 241 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby KG PC ( 1799 -- 1869 ) 28 June 1866 25 February 1868 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli III 7002608000000000000 ♠ 1 year and 243 days The Right Honourable Benjamin Disraeli MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 -- 7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and 77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April 1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone II 23 April 1880 9 June 1885 1880 7003187400000000000 ♠ 5 years and 48 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 23 June 1885 28 January 1886 -- Sec. of State for Foreign Affairs Leader of the House of Lords Conservative Salisbury I 7002220000000000000 ♠ 220 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone III 1 February 1886 20 July 1886 1885 7002170000000000000 ♠ 170 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 July 1886 11 August 1892 1886 Leader of the House of Lords First Lord of the Treasury Sec. of State for Foreign Affairs Conservative Salisbury II 7003221000000000000 ♠ 6 years and 18 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone IV 15 August 1892 2 March 1894 1892 7002565000000000000 ♠ 1 year and 200 days The Right Honourable Archibald Primrose 5th Earl of Rosebery KG PC FRS ( 1847 -- 1929 ) 5 March 1894 22 June 1895 -- First Lord of the Treasury Lord President of the Council Leader of the House of Lords Liberal Rosebery 7002475000000000000 ♠ 1 year and 110 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 June 1895 11 July 1902 Leader of the House of Lords Sec. of State for Foreign Affairs Lord Keeper of the Privy Seal Conservative Salisbury III ( Con. -- Lib. U . ) 1895 1900 7003257300000000000 ♠ 7 years and 17 days Edward VII ( 1901 -- 1910 ) The Right Honourable Arthur Balfour FRS DL MP for Manchester East ( 1848 -- 1930 ) 11 July 1902 5 December 1905 -- First Lord of the Treasury Leader of the House of Commons Conservative Balfour 7003124400000000000 ♠ 3 years and 148 days The Right Honourable Sir Henry Campbell - Bannerman GCB MP for Stirling Burghs ( 1836 -- 1908 ) 5 December 1905 7 April 1908 First Lord of the Treasury Leader of the House of Commons Liberal Campbell - Bannerman 1906 7002855000000000000 ♠ 2 years and 125 days The Right Honourable Herbert Henry Asquith KC FRS MP for East Fife ( 1852 -- 1928 ) 7 April 1908 5 December 1916 First Lord of the Treasury Leader of the House of Commons Sec. of State for War Liberal Asquith I 1910 ​ 1910 George V ( 1910 -- 1936 ) ​ Asquith II ( Lib. -- Con. -- et al . ) 7003316500000000000 ♠ 8 years and 243 days The Right Honourable David Lloyd George OM MP for Caernarvon Boroughs ( 1863 -- 1945 ) 6 December 1916 19 October 1922 First Lord of the Treasury Liberal Lloyd George War 1918 Lloyd George II ( Lib. -- Con . ) 7003214400000000000 ♠ 5 years and 318 days The Right Honourable Bonar Law MP for Glasgow Central ( 1858 -- 1923 ) 23 October 1922 20 May 1923 1922 First Lord of the Treasury Leader of the House of Commons Conservative ( Scot. U . ) Law 7002210000000000000 ♠ 210 days The Right Honourable Stanley Baldwin JP MP for Bewdley ( 1867 -- 1947 ) 23 May 1923 16 January 1924 -- First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Conservative Baldwin I 7002239000000000000 ♠ 239 days The Right Honourable Ramsay MacDonald MP for Aberavon ( 1866 -- 1937 ) 22 January 1924 4 November 1924 1923 First Lord of the Treasury Leader of the House of Commons Sec. of State for Foreign Affairs Labour MacDonald I 7002288000000000000 ♠ 288 days The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 4 November 1924 5 June 1929 1924 First Lord of the Treasury Leader of the House of Commons Conservative Baldwin II 7003167500000000000 ♠ 4 years and 214 days The Right Honourable Ramsay MacDonald FRS MP for Seaham ( 1866 -- 1937 ) 5 June 1929 7 June 1935 1929 First Lord of the Treasury Leader of the House of Commons Labour MacDonald II National Labour National I ( N. Lab. -- Con. -- et al . ) 1931 National II 7003219400000000000 ♠ 6 years and 3 days ​ The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 7 June 1935 28 May 1937 1935 First Lord of the Treasury Leader of the House of Commons Conservative National III ( Con. -- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​ Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26 July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker ( Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26 July 1945 26 October 1951 1945 First Lord of the Treasury Minister of Defence Labour Attlee I 1950 Attlee II 7003228400000000000 ♠ 6 years and 93 days The Right Honourable Sir Winston Churchill KG OM CH TD PCc DL FRS RA MP for Woodford ( 1874 -- 1965 ) 26 October 1951 6 April 1955 1951 First Lord of the Treasury Minister of Defence Conservative Churchill III 7003125900000000000 ♠ 3 years and 163 days Elizabeth II ( 1952 -- present ) The Right Honourable Sir Anthony Eden KG MC MP for Warwick & Leamington ( 1897 -- 1977 ) 6 April 1955 10 January 1957 1955 First Lord of the Treasury Conservative Eden 7002646000000000000 ♠ 1 year and 280 days The Right Honourable Harold Macmillan FRS MP for Bromley ( 1894 -- 1986 ) 10 January 1957 19 October 1963 1959 First Lord of the Treasury Conservative Macmillan I ​ Macmillan II 7003247400000000000 ♠ 6 years and 283 days The Right Honourable Sir Alec Douglas - Home KT MP for Kinross & Western Perthshire ( 1903 -- 1995 ) 19 October 1963 16 October 1964 -- First Lord of the Treasury Conservative ( Scot. U . ) Douglas - Home 7002364000000000000 ♠ 364 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 16 October 1964 19 June 1970 1964 First Lord of the Treasury Minister for the Civil Service Labour Wilson I 1966 Wilson II 7003207300000000000 ♠ 5 years and 247 days The Right Honourable Edward Heath MBE MP for Bexley -- Sidcup ( 1916 -- 2005 ) 19 June 1970 4 March 1970 First Lord of the Treasury Minister for the Civil Service Conservative Heath 7003135500000000000 ♠ 3 years and 259 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 4 March 5 April 1976 First Lord of the Treasury Minister for the Civil Service Labour Wilson III Wilson IV 7002764000000000000 ♠ 2 years and 33 days The Right Honourable James Callaghan MP for Cardiff South East ( 1912 -- 2005 ) 5 April 1976 4 May 1979 -- First Lord of the Treasury Minister for the Civil Service Labour Callaghan 7003112500000000000 ♠ 3 years and 30 days The Right Honourable Margaret Thatcher FRS FRIC MP for Finchley ( 1925 -- 2013 ) See also § Premierships :3 First Lord of the Treasury Minister for the Civil Service Conservative Thatcher I 4 May 1979 28 November 1990 1979 Thatcher II Thatcher III 7003422700000000000 ♠ 11 years and 209 days The Right Honourable John Major MP for Huntingdon ( born 1943 ) 28 November 1990 2 May 1997 First Lord of the Treasury Minister for the Civil Service Conservative Major I ​ Major II 7003234800000000000 ♠ 6 years and 156 days The Right Honourable Tony Blair MP for Sedgefield ( born 1953 ) See also § Premierships : 4 First Lord of the Treasury Minister for the Civil Service Labour Blair I 2 May 1997 27 June 2007 1997 Blair II 2005 Blair III 7003370900000000000 ♠ 10 years and 57 days The Right Honourable Gordon Brown MP for Kirkcaldy & Cowdenbeath ( born 1951 ) See also § Premierships : 5 First Lord of the Treasury Minister for the Civil Service Labour Brown 27 June 2007 11 May -- 7003105000000000000 ♠ 2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July 2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days
  Fact 0.2: Robert Walpole was a Whig.

Retrieved items:
<item 1>
List of Prime Ministers of the United Kingdom
V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name (
Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert
Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the
Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741
7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743
-- First Lord of the Treasury Whig Carteret
</item 1>

<item 2>
List of Prime Ministers of the United Kingdom
^ 7 Majority Conservative government before dissolution . See also ( edit ) United Kingdom This article is part of a series on the politics and government of
the United Kingdom Constitution ( show ) Constitution Cabinet Manual Erskine May Taxation The Crown ( show ) British Monarchy Monarch Queen Elizabeth II Heir
apparent The Prince of Wales Royal family Succession to the British throne Royal prerogative Executive ( show ) HM Government Prime Minister Theresa May Cabinet
Current ministry Her Majesty 's Civil Service Departments Executive agencies Public bodies Privy Council Crown - in - Council Queen Elizabeth II Legislature (
show ) Parliament ( 56th Parliament ) Crown - in - Parliament Queen Elizabeth II House of Lords Lord Speaker The Lord Fowler House of Commons Speaker John
Bercow Leader of the Opposition Jeremy Corbyn Her Majesty 's Loyal Opposition Shadow Cabinet State Opening of Parliament Prime Minister 's Questions Act of
Parliament Statutory Instruments Judiciary ( show ) Supreme Court President The Baroness Hale of Richmond Deputy President The Lord Mance Judges of the Supreme
Court Courts of the United Kingdom Courts of England and Wales Courts of Scotland Courts of Northern Ireland Devolution ( show ) Joint Ministerial Committee
Scotland Government Parliament Wales Government National Assembly Reserved matters Northern Ireland Executive Assembly Administrative geography ( show )
</item 2>

<item 3>
List of Prime Ministers of the United Kingdom
List of Prime Ministers of the United Kingdom - wikipedia List of Prime Ministers of the United Kingdom Jump to : navigation , search Top left : Walpole became
the first Prime Minister of the Kingdom of Great Britain in 1721 . Top right : Pitt the Younger became the first Prime Minister of the United Kingdom of Great
Britain and Ireland in 1801 . Bottom left : Lloyd George was the last Prime Minister of the United Kingdom of Great Britain and Ireland as a result of the Anglo
- Irish Treaty in 1922 . Bottom right : Baldwin was Prime Minister when the country was renamed the United Kingdom of Great Britain and Northern Ireland in 1927
. The Prime Minister of the United Kingdom is the head of the Government of the United Kingdom , and chairs Cabinet meetings . There is no specific date when
the office of Prime Minister first appeared , as the role was not created but rather evolved over a period of time . The term was used in the House of Commons
in 1805 and it was certainly in parliamentary use by the 1880s , and in 1905 the post of Prime Minister was officially given recognition in the order of
precedence .
</item 3>

<item 4>
List of Prime Ministers of the United Kingdom
-- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 --
7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the
Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and
77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord
of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April
1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William
</item 4>

<item 5>
List of Prime Ministers of the United Kingdom
14 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 17 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 28 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 34 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 36 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 42 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 46 . Jump up
^ Eccleshall & Walker ( 2002 ) , p. 54 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 61 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 64 . Jump up ^ Eccleshall
& Walker ( 2002 ) , p. 73 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 77 .
</item 5>

<item 6>
List of Prime Ministers of the United Kingdom
77 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 85 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 94 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 98 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 101 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 106 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 116 . Jump up ^
Eccleshall & Walker ( 2002 ) , p. 120 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 123 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 128 . ^ Jump up to :
Eccleshall & Walker ( 2002 ) , p. 136 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 142 .
</item 6>

<item 7>
List of Prime Ministers of the United Kingdom
BBC News . 13 July 2016 . Retrieved 26 July 2016 . Lay summary ( 13 July 2016 ) . Bibliography ( edit ) Burt , Llewellyn C. ( 1874 ) . A synoptical history of
England ( 2nd ed . ) . London : Lockwood . Carpenter , Clive , ed. ( 1992 ) . The Guinness UK Data Book . Guinness . ISBN 978 - 0 - 85112 - 522 - 0 . Chapman ,
Richard A. ( 31 January 2002 ) . `` History : from earliest times to the present day '' . The Treasury in Public Policy - Making . Routledge . ISBN 978 - 1 -
134 - 86426 - 3 . Clarke , John ( 1999 ) . `` House of Hanover '' . In Fraser , Antonia . The Lives of the Kings & Queens of England . University of California
Press ( published February 2000 ) .
</item 7>

<item 8>
List of Prime Ministers of the United Kingdom
After the succession of King George I in 1714 , the arrangement of a commission of Lords of the Treasury as opposed to a single Lord High Treasurer became
permanent . For the next three years , the government was headed by Lord Townshend , who was appointed Secretary of State for the Northern Department .
Subsequently , Lord Stanhope and Lord Sunderland ran the government jointly , with Stanhope managing foreign affairs and Sunderland managing domestic affairs .
Stanhope died in February 1721 and Sunderland resigned two months later ; Townshend and Robert Walpole were then invited to form the next government . From that
point , the holder of the office of First Lord also unofficially held the status of Prime Minister . It was not until the Edwardian era that the title Prime
Minister was constitutionally recognised . With only few exceptions , the Prime Minister still holds the office of First Lord by constitutional convention .
Since 1721 ( edit ) By century and monarch 18th to 19th century George I George II George III George IV William IV Victoria 20th to 21st century Edward VII
George V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative (
</item 8>

<item 9>
List of Prime Ministers of the United Kingdom
2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the
Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable
Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July
2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days Addenda ( edit ) ^ 1 Whig majority in the Commons before the dissolution of Parliament . ^
2 Minority government before dissolution . ^ 3 Tory majority in the Commons before dissolution . ^ 4 Majority Tory government before dissolution . ^ 5 Majority
Liberal government before dissolution . ^ 6 Coalition government during the course of an undissolved hung parliament . ^ 7 Majority Conservative government
before dissolution . See
</item 9>

<item 10>
List of Prime Ministers of the United Kingdom
-- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS
MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​
Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26
July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker (
Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26
July
</item 10>
=== END ITEM id=a185-97242bf8a40f049f ===

=== ITEM id=a173-18e3e4005cc7fe39 ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.

Retrieved items:
<item 1>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 1>

<item 2>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 2>

<item 3>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 3>

<item 4>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 4>

<item 5>
Beijing
Two generations later , Kublai Khan ordered the construction of Dadu ( or Daidu to the Mongols , commonly known as Khanbaliq ) , a new capital for his Yuan
dynasty to the northeast of the Zhongdu ruins . The construction took from 1264 to 1293 , but greatly enhanced the status of a city on the northern fringe of
China proper . The city was centered on the Drum Tower slightly to the north of modern Beijing and stretched from the present - day Chang'an Avenue to the
northern part of Line 10 subway . Remnants of the Yuan rammed earth wall still stand and are known as the Tucheng . Ming dynasty One of the corner towers of the
Forbidden City , which was built by Emperor Yongle in Ming dynasty . In 1368 , soon after declaring the new Hongwu era of the Ming dynasty , the rebel leader
Zhu Yuanzhang sent an army to Khanbaliq and conquered it . Since the Yuan continued to occupy Shangdu and Mongolia , however , Dadu was renamed to Beiping as
used to supply the military garrisons in the area .
</item 5>

<item 6>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 6>

<item 7>
Beijing
On 28 June the same year , Beijing 's name was returned to Beiping ( written at the time as `` Peiping '' ) . On July 7 , 1937 , the 29th Army and the Japanese
army in China exchanged fire at the Marco Polo Bridge near the Wanping Fortress southwest of the city . The Marco Polo Bridge Incident triggered the Second Sino
- Japanese War , World War II as it is known in China . During the war , Beiping fell to Japan on 29 July 1937 and was made the seat of the Provisional
Government of the Republic of China , a puppet state that ruled the ethnic - Chinese portions of Japanese - occupied northern China . This government was later
merged into the larger Wang Jingwei government based in Nanjing . People 's Republic of China Mao Zedong proclaiming the establishment of the People 's Republic
of China in 1949 In the final phases of the Chinese Civil War , the People 's Liberation Army seized control of the city peacefully on 31 January 1949 in the
course of the Pingjin Campaign .
</item 7>

<item 8>
Beijing
In 938 , after the fall of the Tang , the Later Jin ceded the entire northern frontier to the Khitan Liao dynasty , which treated the city as Nanjing , or the
`` Southern Capital '' , one of four secondary capitals to complement its `` Supreme Capital '' , Shangjing ( modern Baarin Left Banner in Inner Mongolia ) .
Some of the oldest surviving structures in Beijing date to the Liao period , including the Tianning Pagoda . Miaoying Temple and its White Stupa in Dadu which
was built by Kublai Khan The Liao fell to the Jurchen Jin dynasty in 1122 , which gave the city to the Song dynasty and then retook it in 1125 during its
conquest of northern China . In 1153 , the Jurchen Jin made Beijing their `` Central Capital '' , or Zhongdu . The city was besieged by Genghis Khan 's invading
Mongolian army in 1213 and razed to the ground two years later .
</item 8>

<item 9>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 9>

<item 10>
Beijing
After the First Emperor unified China , Jicheng became a prefectural capital for the region . During the Three Kingdoms period , it was held by Gongsun Zan and
Yuan Shao before falling to Cao Cao 's Wei Kingdom . The AD 3rd - century Western Jin demoted the town , placing the prefectural seat in neighboring Zhuozhou .
During the Sixteen Kingdoms period when northern China was conquered and divided by the Wu Hu , Jicheng was briefly the capital of the Xianbei Former Yan
Kingdom . After China was reunified during the Sui dynasty , Jicheng , also known as Zhuojun , became the northern terminus of the Grand Canal . Under the Tang
dynasty , Jicheng as Youzhou , served as a military frontier command center . During the An - Shi Rebellion and again amidst the turmoil of the late Tang ,
local military commanders founded their own short - lived Yan dynasties and called the city Yanjing , or the `` Yan Capital . '' Also in the Tang dynasty , the
city 's name Jicheng was replaced by Youzhou or Yanjing .
</item 10>
=== END ITEM id=a173-18e3e4005cc7fe39 ===

=== ITEM id=a172-3cd24c8a093ce350 ===
Gold sentences and facts:
Sentence 0: Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20 nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond Canada ← 2014 2022 →
  Fact 0.5: Alina Zagitova of Olympic Athletes from Russia was the first-place medalist.

Retrieved items:
<item 1>
Figure skating at the 2018 Winter Olympics – Ladies' singles
21 49.82 50.39 6.32 6.29 6.21 6.39 6.29 - 3.00 6 24 Isadora Williams Brazil 88.44 38.39 51.05 6.43 6.18 6.36 6.43 6.50 - 1.00 12 * Evgenia Medvedeva finished at
higher place due to better program component score Overall ( edit ) The skaters were ranked according to their overall score . TP - Total points ; SP - Short
program ; FS - Free skating Rank Name Nation Total points SP FS 01 ! Alina Zagitova Olympic Athletes from Russia 239.57 82.92 156.65 02 ! Evgenia Medvedeva
Olympic Athletes from Russia 238.26 81.61 156.65 03 ! Kaetlyn Osmond Canada 231.02 78.87 152.15 Satoko Miyahara Japan 222.38 75.94 146.44 5 Carolina Kostner
Italy 212.
</item 1>

<item 2>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics . Retrieved 26 December 2017 . ^ Jump up to : `` Schedule '' . Retrieved 6 January
2018 . Jump up ^ Start list ^ Jump up to : Bieler , Des ; Janes , Chelsea ( 21 February 2018 ) . `` Alina Zagitova tops Russian teammate Evgenia Medvedeva 's
world - record short program score '' . The Washington Post . ISSN 0190 - 8286 . Retrieved 21 February 2018 . ^ Jump up to : `` Russia 's Zagitova beats
Medvedeva 's short program world record at Olympics '' . TASS. 21 February 2018 . Retrieved 21 February 2018 . Jump up ^ `` Qualification Systems for XXIII
Olympic Winter Games , PyeongChang 2018 , Figure skating '' . International Skating Union ( ISU ) . 17 May 2016 . Retrieved 20 January 2018 .
</item 2>

<item 3>
Figure skating at the 2018 Winter Olympics – Ladies' singles
24 quotas were handed out during the 2017 World Figure Skating Championships and the remaining six were given out at the 2017 CS Nebelhorn Trophy . Each country
decided the entry of its teams , and athletes winning the quota were not necessarily granted the right to compete . All athletes competing must have met the
minimum total elements score , which does not include component scores . For the short program this was 20.00 and the free skate 36.00 . Schedule ( edit ) All
times are ( UTC + 9 ) . Date Time Round 21 February 10 : 00 Short program 23 February 10 : 00 Free skating Results ( edit ) Short program ( edit ) The short
program was held on 21 February . Pl . Name Nation TSS TES PCS SS TR PE CH IN Ded StN Alina Zagitova Olympic Athletes from Russia 82.92 WR 45.30 37.62 9.36 9.18
9.64 9.43 9.43 0.00 28 Evgenia Medvedeva Olympic Athletes from Russia 81.61 43.
</item 3>

<item 4>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Figure skating at the 2018 Winter Olympics -- ladies ' singles - wikipedia Figure skating at the 2018 Winter Olympics -- ladies ' singles Jump to : navigation ,
search Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20
nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond
Canada ← 2014 2022 → Figure skating at the 2018 Winter Olympics Qualification Singles men ladies Pairs mixed Ice dancing mixed Team event mixed The ladies '
single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21
February , and the free skating was held on 23 February .
</item 4>

<item 5>
Figure skating at the 2018 Winter Olympics – Ladies' singles
97 20 23 Ivett Tóth Hungary 150.43 53.22 23 97.21 23 24 Isadora Williams Brazil 144.18 55.74 17 88.44 24 Did not advance to free skating 25 Larkyn Austman
Canada N / A 51.42 25 N / A N / A 26 Diāna Ņikitina Latvia N / A 51.12 26 N / A N / A 27 Giada Russo Italy N / A 50.88 27 N / A N / A 28 Anita Östlund Sweden N
/ A 49.14 28 N / A N / A 29 Anna Khnychenkova Ukraine N / A 47.59 29 N / A N / A 30 Aiza Mambekova Kazakhstan N / A 44.40 30 N / A N / A References ( edit )
Jump up ^ `` Venues '' . www.pyeongchang2018.com/ . Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics .
</item 5>

<item 6>
Figure skating at the 2018 Winter Olympics – Ladies' singles
33 21 121.38 10 14 Nicole Rajičová Slovakia 175.19 60.59 13 114.60 15 15 Gabrielle Daleman Canada 172.46 68.90 7 103.56 19 16 Loena Hendrickx Belgium 171.88
55.16 20 116.72 14 17 Kailani Craine Australia 168.61 56.77 16 111.84 16 18 Nicole Schott Germany 168.46 59.20 14 109.26 17 19 Maé - Bérénice Méité France
159.92 53.67 22 106.25 18 20 Emmi Peltonen Finland 157.14 55.28 18 101.86 21 21 Alexia Paganini Switzerland 156.26 55.26 19 101.00 22 22 Li Xiangning China
154.43 52.46 24 101.97 20 23 Ivett Tóth Hungary 150.43 53.
</item 6>

<item 7>
Figure skating at the 2018 Winter Olympics – Ladies' singles
TES - Technical Element Score ; PCS - Program Component Score SS - Skating Skills ; TR - Transitions ; PE - Performance / Execution CH - Choreography ; IN -
Interpretation ; Ded - Deduction ; StN - Starting Number Free skating ( edit ) The free skating was held on 23 February . Pl . Name Nation TSS TES PCS SS TR PE
CH IN Ded StN Evgenia Medvedeva Olympic Athletes from Russia * 156.65 * 79.18 * 77.47 * 9.57 9.43 9.82 9.71 9.89 0.00 24 Alina Zagitova Olympic Athletes from
Russia * 156.65 * 81.62 * 75.03 * 9.32 9.21 9.61 9.29 9.46 0.00 22 Kaetlyn Osmond Canada 152.15 76.50 75.65 9.50 9.21 9.57 9.39 9.61 0.00 23 Satoko Miyahara
Japan 146.
</item 7>

<item 8>
Figure skating at the 2018 Winter Olympics – Ladies' singles
38 75.94 146.44 5 Carolina Kostner Italy 212.44 73.15 6 139.29 5 6 Kaori Sakamoto Japan 209.71 73.18 5 136.53 6 7 Choi Da - bin South Korea 199.26 67.77 8
131.49 8 8 Maria Sotskova Olympic Athletes from Russia 198.10 63.86 12 134.24 7 9 Bradie Tennell United States 192.35 64.01 11 128.34 9 10 Mirai Nagasu United
States 186.54 66.93 9 119.61 12 11 Karen Chen United States 185.65 65.90 10 119.75 11 12 Elizabet Tursynbayeva Kazakhstan 177.12 58.82 15 118.30 13 13 Kim Ha -
nul South Korea 175.71 54.33 21 121.38 10 14 Nicole Rajičová Slovakia 175.
</item 8>

<item 9>
Figure skating at the 2018 Winter Olympics – Ladies' singles
17 May 2016 . Retrieved 20 January 2018 . Jump up ^ `` Extract from ISU Communication 2097 '' . www.iceskating.org.uk/ . National Ice Skating Association ( UK )
. Retrieved 20 January 2018 . Jump up ^ Short program results Jump up ^ Free skating results Jump up ^ Final results Olympic champions in figure skating --
Ladies ' singles 1908 : Madge Syers ( GBR ) 1920 : Magda Julin ( SWE ) 1924 : Herma Szabo ( AUT ) 1928 : Sonja Henie ( NOR ) 1932 : Sonja Henie ( NOR ) 1936 :
Sonja Henie ( NOR ) 1948 : Barbara Ann Scott ( CAN ) 1952 : Jeannette Altwegg ( GBR ) 1956 : Tenley Albright ( USA ) 1960 : Carol Heiss ( USA ) 1964 : Sjoukje
Dijkstra ( NED ) 1968 : Peggy Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy
</item 9>

<item 10>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy Hamill ( USA ) 1980 : Anett Pötzsch ( GDR ) 1984 : Katarina Witt ( GDR ) 1988 : Katarina Witt ( GDR )
1992 : Kristi Yamaguchi ( USA ) 1994 : Oksana Baiul ( UKR ) 1998 : Tara Lipinski ( USA ) 2002 : Sarah Hughes ( USA ) 2006 : Shizuka Arakawa ( JPN ) 2010 : Yuna
Kim ( KOR ) 2014 : Adelina Sotnikova ( RUS ) 2018 : Alina Zagitova ( OAR ) Record scores in figure skating Combined total ( Total ) Short program ( SP ) Free
skating ( FS ) Short dance ( SD ) Free dance ( FD ) Juniors ( J ) Men Total : Yuzuru Hanyu 330.43 SP : Yuzuru Hanyu 112.72 FS : Yuzuru Hanyu 223.
</item 10>
=== END ITEM id=a172-3cd24c8a093ce350 ===

=== ITEM id=a159-706ca1df308f9f92 ===
Gold sentences and facts:
Sentence 0: In probability theory , two events are independent , statistically independent , or stochastically independent if the occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does not affect the probability distribution of the other .
  Fact 0.1: In probability theory, two events can be called statistically independent.

Retrieved items:
<item 1>
Independence (probability theory)
A finite set of events is mutually independent if every event is independent of any intersection of the other events -- that is , if and only if for every k -
element subset of ( A i ) i = 1 n ( \ displaystyle \ ( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) , P ( ⋂ i = 1 k A i ) = ∏ i = 1 k P ( A i ) . ( \ displaystyle \
mathrm ( P ) \ left ( \ bigcap _ ( i = 1 ) ^ ( k ) A_ ( i ) \ right ) = \ prod _ ( i = 1 ) ^ ( k ) \ mathrm ( P ) ( A_ ( i ) ) . ) This is called the
multiplication rule for independent events . Note that it is not a single condition involving only the product of all the probabilities of all single events (
see below for a counterexample ) ; it must hold true for all subsets of events .
</item 1>

<item 2>
Independence (probability theory)
) and yet no two of the three events are pairwise independent ( and hence the set of events are not mutually independent ) . This example shows that mutual
independence involves requirements on the products of probabilities of all combinations of events , not just the single events as in this example . See also (
edit ) Copula ( statistics ) Independent and identically distributed random variables Mutually exclusive events Subindependence Linear dependence between random
variables Conditional independence Normally distributed and uncorrelated does not imply independent Mean dependence References ( edit ) Jump up ^ Russell ,
Stuart ; Norvig , Peter ( 2002 ) . Artificial Intelligence : A Modern Approach . Prentice Hall . p. 478 . ISBN 0 - 13 - 790395 - 2 . ^ Jump up to : Feller , W (
1971 ) . `` Stochastic Independence '' . An Introduction to Probability Theory and Its Applications . Wiley . Jump up ^ George , Glyn , `` Testing for the
independence of three events , '' Mathematical Gazette 88 , November 2004 , 568 .
</item 2>

<item 3>
Independence (probability theory)
Thus , the occurrence of B does not affect the probability of A , and vice versa . Although the derived expressions may seem more intuitive , they are not the
preferred definition , as the conditional probabilities may be undefined if P ( A ) or P ( B ) are 0 . Furthermore , the preferred definition makes clear by
symmetry that when A is independent of B , B is also independent of A . More than two events ( edit ) A finite set of events ( A i ) i = 1 n ( \ displaystyle \
( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) is pairwise independent if every pair of events is independent -- that is , if and only if for all distinct pairs of
indices m , k , P ( A m ∩ A k ) = P ( A m ) P ( A k ) . ( \ displaystyle \ mathrm ( P ) ( A_ ( m ) \ cap A_ ( k ) ) = \ mathrm ( P ) ( A_ ( m ) ) \ mathrm ( P )
( A_ ( k ) ) . )
</item 3>

<item 4>
Independence (probability theory)
More than two random variables ( edit ) A set of random variables is pairwise independent if and only if every pair of random variables is independent . Even if
the set of random variables is pairwise independent , it is not necessarily mutually independent as defined next . A set of random variables is mutually
independent if and only if for any finite subset X 1 , ... , X n ( \ displaystyle X_ ( 1 ) , \ ldots , X_ ( n ) ) and any finite sequence of numbers a 1 , ... ,
a n ( \ displaystyle a_ ( 1 ) , \ ldots , a_ ( n ) ) , the events ( X 1 ≤ a 1 ) , ... , ( X n ≤ a n ) ( \ displaystyle \ ( X_ ( 1 ) \ leq a_ ( 1 ) \ ) , \ ldots
, \ ( X_ ( n ) \ leq a_ ( n ) \ ) ) are mutually independent events ( as defined above ) . The measure - theoretically inclined may prefer to substitute events
( X ∈ A ) for events ( X ≤ a ) in the above definition , where A is any Borel set .
</item 4>

<item 5>
Independence means one event’s occurrence does not change another’s probability: \(P(A\cap B)=P(A)P(B)\). For random variables, their joint distribution factors
into their marginal distributions. With multiple events or variables, pairwise independence does not necessarily imply mutual independence; mutual independence
requires the product rule for every subset. Conditional independence means variables are independent once another variable is known. Independent variables have
zero covariance, but zero covariance alone does not imply independence. Examples: separate die rolls are independent
</item 5>

<item 6>
Independence (probability theory)
Independence ( probability theory ) - wikipedia Independence ( probability theory ) Jump to : navigation , search Part of a series on Statistics Probability
theory Probability axioms Probability space Sample space Elementary event Event Random variable Probability measure Complementary event Joint probability
Marginal probability Conditional probability Independence Conditional independence Law of total probability Law of large numbers Bayes ' theorem Boole 's
inequality Venn diagram Tree diagram In probability theory , two events are independent , statistically independent , or stochastically independent if the
occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does
not affect the probability distribution of the other . The concept of independence extends to dealing with collections of more than two events or random
variables , in which case the events are pairwise independent if each pair are independent of each other , and the events are mutually independent if each event
is independent of each other combination of events . Contents ( hide ) 1 Definition 1.1 For events 1.1. 1 Two events 1.1. 2 More than two events 1.2 For random
variables 1.2. 1 Two random variables 1.2.
</item 6>

<item 7>
Independence (probability theory)
Drawing cards ( edit ) If two cards are drawn with replacement from a deck of cards , the event of drawing a red card on the first trial and that of drawing a
red card on the second trial are independent . By contrast , if two cards are drawn without replacement from a deck of cards , the event of drawing a red card
on the first trial and that of drawing a red card on the second trial are not independent , because a deck that has had a red card removed has proportionately
fewer red cards . Pairwise and mutual independence ( edit ) Pairwise independent , but not mutually independent , events . Mutually independent events .
Consider the two probability spaces shown . In both cases , P ( A ) = P ( B ) = 1 / 2 and P ( C ) = 1 / 4 .
</item 7>

<item 8>
Independence (probability theory)
A ( \ displaystyle ( \ mathcal ( A ) ) ) and B ( \ displaystyle ( \ mathcal ( B ) ) ) are said to be independent if , whenever A ∈ A ( \ displaystyle A \ in ( \
mathcal ( A ) ) ) and B ∈ B ( \ displaystyle B \ in ( \ mathcal ( B ) ) ) , P ( A ∩ B ) = P ( A ) P ( B ) . ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \
mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) . )
</item 8>

<item 9>
Independence (probability theory)
1 Two random variables 1.2. 2 More than two random variables 1.2. 3 Conditional independence 1.3 Independent σ - algebras 2 Properties 2.1 Self - independence
2.2 Expectation and covariance 2.3 Characteristic function 3 Examples 3.1 Rolling dice 3.2 Drawing cards 3.3 Pairwise and mutual independence 3.4 Mutual
independence 4 See also 5 References Definition ( edit ) For events ( edit ) Two events ( edit ) Two events A and B are independent ( often written as A ⊥ B ( \
displaystyle A \ perp B ) or A ⊥ ⊥ B ( \ displaystyle A \ perp \ ! \ ! \ ! \ perp B ) ) if their joint probability equals the product of their probabilities : P
( A ∩ B ) = P ( A ) P ( B ) ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \ mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) ) .
</item 9>

<item 10>
Independence (probability theory)
For more than two events , a mutually independent set of events is ( by definition ) pairwise independent ; but the converse is not necessarily true ( see below
for a counterexample ) . For random variables ( edit ) Two random variables ( edit ) Two random variables X and Y are independent if and only if ( iff ) the
elements of the π - system generated by them are independent ; that is to say , for every a and b , the events ( X ≤ a ) and ( Y ≤ b ) are independent events (
as defined above ) .
</item 10>
=== END ITEM id=a159-706ca1df308f9f92 ===

=== ITEM id=a196-2f9a101b9393f0cc ===
Gold sentences and facts:
Sentence 0: Bridgestone / Firestone also produces tires under the Fuzion brand name . Fuzion is a tire manufactured in China for Bridgestone / Firestone as a budget replacement tire .
  Fact 1.1: Fuzion is a tire manufactured in China for Bridgestone / Firestone.

Retrieved items:
<item 1>
Firestone Tire and Rubber Company
This included both general price and foreign exchange controls . When the 10 - year period came to an end in 1979 , Firestone retaliated by increasing
production , making entry less attractive . Headquarters eventually canceled expansion and failed negotiations lead to no further investments . In motorsport (
edit ) The 1911 Indianapolis 500 auto race was won by a car running Firestone tires . Firestone - shod cars won all editions of the race from 1920 to 1966 . The
company also provided tires to Formula One from 1950 to 1974 . Citing rising costs as the cause , Firestone announced its withdrawal from professional
motorsport in August 1974 with the exception of Formula 5000 and Formula One . The change was forecast to reduce Firestone 's annual race testing budget from
between $6 and $8 million dollars to around $500,000 . Shortly afterwards , Firestone extended this to a complete withdrawal by the end of the 1974 season . The
manufacturer returned in 1995 to the CART series with technical assistance from Bridgestone .
</item 1>

<item 2>
Firestone Tire and Rubber Company
Harvey Firestone had a friendship with Henry Ford . Firestone used this relationship to become the original equipment supplier of Ford Motor Company automobiles
, and was also active in the replacement market . In 1988 , the company was sold to the Japanese Bridgestone Corporation . Contents ( hide ) 1 History 1.1 Early
- to - mid-20th century 1.2 Restructuring and sale to Bridgestone 2 In motorsport 3 Corporate troubles 3.1 Great American streetcar scandal 3.2 Firestone 500
tread separation problem 3.3 Liberian rubber plantation 3.4 Ford Explorer rollover problem 4 Television advertisement jingle 5 Other brands 6 See also 7
References 8 External links History ( edit ) Early - to - mid-20th century ( edit ) The First Firestone store Firestone newspaper ad from 1919 , explaining the
benefits of their cord tire . Firestone was originally based in Akron , Ohio , also the hometown of its archrival , Goodyear Tire and Rubber Company , and
another two mid-sized competitors , General Tire and Rubber and BF Goodrich .
</item 2>

<item 3>
Firestone Tire and Rubber Company
East Los Angeles College has proposed a new satellite campus at the site . In 1928 the company built a factory in Brentford , England , a longtime Art Deco
landmark on a major route into the city . This closed in 1979 . In 1936 the company opened a plant in Memphis , Tennessee . With a work force exceeding 3,000
employees , the Memphis plant was the largest tire manufacturer in the company 's worldwide operation . On July 1 , 1963 , the company celebrated the production
of 100 million tires in Memphis . The plant was closed in 1982 . During World War II the company was called on by the U.S. Government to make artillery shells ,
aluminum kegs for food transport and rubberized military products . Firestone ranked 55th among United States corporations in the value of World War II military
production contracts . In the 1940s , Firestone was given a defense contract to produce plastic helmet liners . While outproduced by Westinghouse Electric they
still made a fair amount for the M1 Helmet .
</item 3>

<item 4>
Firestone Tire and Rubber Company
The combined Bridgestone / Firestone North American operations are now based in Nashville , Tennessee . In 2008 , the companies celebrated a 20 - year
anniversary of the merger , and changed the tire division name to Bridgestone Americas Tire Operations , LLC . In 2012 , Bridgestone Americas opened a $100
million technical center in Akron . Firestone tires used for the Indianapolis 500 race . Apart from tires , several companies and divisions operate with the
Firestone brand in its name . These companies include Firestone Building Products , LLC , Firestone Industrial Products , LLC , Firestone Complete Auto Care ,
Firestone Natural Rubber Company and Firestone Specialty Products . Firestone Building Products and Firestone Industrial Products are headquartered in
Indianapolis , Indiana and operate in 23 U.S. states and ten countries , with 11,000 employees worldwide . The company reports annual sales of more than $2.5
billion . In 1972 , Firestone received a ten - year import `` concession '' by the Kenyan government to secure Firestone 's investment in a domestic tire plant
, which gave it a virtual monopoly . This included both general price and foreign exchange controls .
</item 4>

<item 5>
Firestone Tire and Rubber Company
Founded on August 3 , 1900 , the company initiated operations with 12 employees . Together , Firestone and Goodyear were the largest suppliers of automotive
tires in North America for over 75 years . In 1906 Henry Ford chose Firestone for Model T original equipment tires . In 1918 , Firestone Tire and Rubber Company
of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone
produced the Oldfield tire , named for racing driver Barney Oldfield . In 1926 , the company opened one of the world 's biggest rubber plantations in Liberia ,
West Africa , spanning more than 1 million acres . 1926 was also the year that the company opened its first Firestone Complete Auto Care store ( Firestone
Complete Auto Care is a division of Firestone that offers automotive maintenance and repair ) . In 1927 , Henry Ford and tire maven Harvey Firestone took a trip
to Los Angeles to select locations for their new factories .
</item 5>

<item 6>
Firestone Tire and Rubber Company
Firestone tire and Rubber company - wikipedia Firestone tire and Rubber company Jump to : navigation , search Firestone Tire and Rubber Company Type Subsidiary
Industry Manufacturing Founded Akron , Ohio , United States ( August 3 , 1900 ; 117 years ago ( 1900 - 08 - 03 ) ) Founder Harvey Firestone Headquarters 200 4th
Ave S , Nashville , Tennessee , United States Area served Worldwide Key people Gordon Knapp ( CEO and President ) Bill Thompson ( COO ) Products Tires Revenue
US $ 2.09 billion ( 2004 ) Number of employees 40,000 Parent Bridgestone ( 1988 -- present ) Website firestone.com The Firestone Tire and Rubber Company is an
American tire company founded by Harvey Samuel Firestone in 1900 to supply pneumatic tires for wagons , buggies , and other forms of wheeled transportation
common in the era . Firestone soon saw the huge potential for marketing tires for automobiles . The company was a pioneer in the mass production of tires .
Harvey Firestone had a friendship with Henry Ford .
</item 6>

<item 7>
Firestone Tire and Rubber Company
ISBN 978 - 0 - 15 - 692620 - 1 . Jump up ^ `` A Global Dream '' . Bridgestone Americas . Retrieved 3 August 2013 . Jump up ^ `` History of Bridgestone /
Firestone '' . Bridgestone / Firestone Canada . 2009 . Archived from the original on 10 May 2010 . Jump up ^ Firestone 8000 Tire Order by Ford Ford Chooses
Firestone for Model T Jump up ^ Canada manufactures first Firestone tire Archived 2010 - 05 - 10 at the Wayback Machine . Firestone Canada Incorporated Jump up
^ Chris Nichols ( 2013 - 03 - 01 ) . `` DispL. A. Case # 52 : L.A. 's First Firestone Tire - Los Angeles Magazine '' . Lamag.com . Retrieved 2015 - 12 - 17 .
Jump up ^ Lost Memphis Firestone Tire & Rubber Company Jump up ^ Firestone Closings Jump up ^ Peck , Merton J.
</item 7>

<item 8>
Firestone Tire and Rubber Company
External links ( edit ) Wikimedia Commons has media related to Firestone Tire and Rubber Company . Official website Century of Progress from the Summit Memory
Project Bridgestone Corporation Subsidiaries Firestone Diversified Products Firestone Tire and Rubber Company Firestone - Apsley Rubber Company Products
Bridgestone Picnica Helmz Facilities Firestone Tyre Factory People Harvey Samuel Firestone Related Akron Firestone Non-Skids Bridgestone Arena Bridgestone
Museum of Art Bridgestone Open Firestone and Ford tire controversy Firestone Country Club International Labor Rights Forum v Firestone Tire and Rubber Co The
Voice of Firestone WGC - Bridgestone Invitational Automotive industry in the United States Automotive industry Economy of the United States Transportation in
the United States American vehicle manufacturers ( list ) AGCO Challenger Tractor Massey Ferguson AM General American Expedition Vehicles American Growler Amp
Electric Vehicles Anteros Coachworks Arcimoto Armour Group , Inc .
</item 8>

<item 9>
Firestone Tire and Rubber Company
Friends say Ford wanted to be near the ocean and picked Long Beach and suggested Firestone go to South Gate , California . The tiny community southeast of
Downtown was mostly agriculture at the time and Firestone found 40 acres of beanfield to house his new manufacturing plant . Architects Curlett and Beelman
created a spectacular four - storey Italianate complex , with its own power plant and gorgeous polychrome murals by Gladding McBean depicting the tire and
rubber - making process . A year after the plant opened in 1928 it doubled in size . By 1954 , when they added the Corporal guided missile to their offerings ,
the plant was nearly a million square feet . The town grew around Firestone , they named the main boulevard through town after Harvey , and Los Angeles became
the number one tire market in the country . By the mid-1970s Ford and GM had massive layoffs as Firestone and other manufacturers opened new plants in non-union
locales like Wilson , North Carolina . After much downsizing the end came in 1980 when 1,300 workers were laid off and the plant closed .
</item 9>

<item 10>
Firestone Tire and Rubber Company
Retrieved 2007 - 03 - 21 . Jump up ^ Isidore , Chris ( May 21 , 2001 ) . `` Firestone severs ties to Ford '' . CNN . Jump up ^ NHTSA Recall Campaign 01T - 016
Q4 Vehicle Safety Recall Quarterly Report Information , January 25 , 2008 ( Amended March 26 , 2009 ) ( from the National Highway Traffic Safety Administration
) ^ Jump up to : Channel 2 Investigation Finds Recalled Tires for Sale , November 4 , 2013 Archived November 5 , 2013 , at the Wayback Machine . Jump up ^ ``
Firestone Jingle '' ( QuickTime ) . Firestone . Retrieved 2007 - 03 - 22 . External links ( edit ) Wikimedia Commons has media related to Firestone Tire and
Rubber Company .
</item 10>
=== END ITEM id=a196-2f9a101b9393f0cc ===

=== ITEM id=a176-d2f6521da5f57b8c ===
Gold sentences and facts:
Sentence 0: No . overall No. in series Title Vehicles tested Special guests Original air date 14 `` Past , Present , or Future '' Lamborghini Aventador S , Honda NSX , Rimac Concept One David Hasselhoff , Ricky Wilson 8 December 2017 ( 2017 - 12 - 08 ) The tent is now in the Cotswolds and will remain there for the duration of the series . To determine whether the past , present , or future of motoring is best , the presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past , May in a hybrid powered Honda NSX representing the present , and Hammond in an all - electric Rimac Concept One representing the future . After starting at the top of the Gotthard Pass , Hammond brings the trio to a stay at the Park Weggis ( de ) health retreat and visit the Swiss Museum of Transport and other museums in Lucerne close to a charging station for the Rimac car . May and Clarkson reject this and go to a hill climbing race at Hemberg instead . As Clarkson can not fit into the Aventador whilst wearing a mandatory racing helmet , Clarkson sends a Lamborghini test driver in his place , thus setting the fastest time of all three . Hammond crashes after completing the course , destroying the Rimac in a fire . Footage taken from YouTube shows the wreckage being extinguished and Hammond being airlifted by the Swiss Air - Rescue ( REGA ) to hospital . The presenters discuss some inapropriate new features within cars on Conversation Street . Celebrity Brain Crash is replaced by Celebrity Face Off , in which Ricky Wilson beats David Hasselhoff to determine who the `` fastest former talent show host '' is . 15 `` The Falls Guys '' Ford GT , Mercedes AMG GT R Kevin Pietersen , Brian Wilson 15 December 2017 ( 2017 - 12 - 15 ) The presenters race from Central Park in Manhattan , New York City , to an observation tower overlooking the Niagara Falls . Clarkson drives the new Ford GT whilst May and Hammond ( on crutches ) use public transport , taking the New York City Subway , AirTrain JFK , a JetBlue Embraer E190 flight , plus two Niagara Frontier Transportation Authority buses . Despite Hammond and May 's flight arriving 15 minutes early , Clarkson narrowly wins the race . For Conversation Street the presenters discuss sponsorship ideas . Clarkson reviews the Mercedes AMG GT R , after which new test driver , Abbie Eaton , takes it for a timed lap around the Eboladrome . In `` Celebrity Face Off '' , Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the `` fastest person who makes a living throwing and hitting and catching small balls '' . 16 `` Bah Humbug - atti '' Kia Stinger GT , Bugatti Chiron Hugh Bonneville , Casey Anderson 22 December 2017 ( 2017 - 12 - 22 ) The presenters begin with Christmas tree lights made from vehicle parts and , Starting with Scalextric , they exchange humorous and inappropriate Christmas gifts in Conversation Street . May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road , with May winning the race despite him racing uphill and the riders downhill . Hammond and May attempt to relieve lunchtime boredom by inventing `` lunchtime office car park racing '' at Houghton Hall business park in the Luton / Dunstable Urban Area . After brief interviews in which both share their own ownership of a Tesla Model S , Hugh Bonneville , representing the UK , and Casey Anderson , representing the United States , compete during `` Celebrity Face Off '' as the `` fastest person who lives with a bear '' , with Anderson winning by setting the faster lap time . Clarkson experiments with the jet set lifestyle , starting in the morning from the Da Vinci super-yacht moored in Saint - Tropez and then driving a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime , stopping briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is . After leaving the Alps , Clarkson then pays an homage to the 1976 film C'était un rendez - vous , by speeding across Turin and reaching the Piazza Carignano ( it ) in 9 minutes . 17 `` Unscripted '' McLaren 720S , Audi TT RS , Ariel Nomad Michael Ball , Alfie Boe , Silvija Jurin 29 December 2017 ( 2017 - 12 - 29 ) The presenters begin by having a year - ending awards ceremony in Conversation Street . Hammond heads to the Eboladrome to test the McLaren 720S after which Eaton sets a lap time . Since The Grand Tour has been criticised for being too `` scripted '' , the presenters head to Zadar county and the island of Pag in Croatia to make a completely `` unscripted '' film . This results in Clarkson turning up in the Audi TT RS , Hammond turning up in the not at all comparable Ariel Nomad , and May turning up in an old Lada Riva which he turns into a fire engine . They go their separate ways though Clarkson and Hammond end up comparing their cars in a drag race , a rally stage ( both of which are won by Clarkson ) , visiting Maslenica , the Plitvice Lakes National Park , and following the D106 road until running out of road at Žigljen ferry port . Due to a lack of fires , May has to setup his own ones until Clarkson and Hammond setup one that proves too difficult for May , which the real fire brigade manage to put out with a firefighting aircraft , destroying May 's Lada fire engine in the process . Michael Ball and Alfie Boe compete in `` Celebrity Face Off '' with Ball setting the faster time and becoming the `` fastest classical singer with some kind of interesting connection to the British motor industry '' . 18 5 `` Up , Down and Round the Farm '' Ripsaw EV2 , Volkswagen Up ! GTI Bill Bailey , Dominic Cooper , Mark Higgins 5 January 2018 ( 2018 - 01 - 05 ) The presenters introduce some of the new cars for 2018 in Conversation Street . Hammond heads to Dubai to test a Sandrail dune buggy in the desert , followed by the Ripsaw `` high - end luxury tank '' , visiting the Dubai Mall . May reviews the Volkswagen Up ! GTI at the Eboladrome and in thick fog Eaton records a record slow time . Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars before racing each other in greasy / wet conditions with Cooper becoming the `` fastest person who has a failed former career in a band '' . Clarkson creates a Ken Block - style `` Farmkhana '' film using an old Subaru Impreza WRX STi around Hare Farm , near Rye , with bullet time photography . Hammond and May show the audience that Clarkson 's driving was mostly fake by sharing a `` making - of '' consisting of outtakes and behind - the - scenes footage featuring rally driver Mark Higgins , producer Phil Churchward , plus camera operator Ben Joiner breaking the fourth wall . Clarkson suggests that the whole thing is like sawing a woman in half during a magic show , and that really the audience do n't want to know . 19 6 `` Jaaaaaaaags '' None Luke Evans , Kiefer Sutherland 12 January 2018 ( 2018 - 01 - 12 ) The presenters head to the area around Grand Junction , Colorado , staying at the Gateway Canyons Resort & Spa , and eventually travelling to Telluride , Colorado with a variety of old Jaguar cars to establish the manufacturer 's reputation for reliability with Clarkson buying an XJR , May taking an XK8 convertible and Hammond bringing a Mark X . During the course of the trip the cars are driven around a dirt track ( where Hammond 's car is badly damaged , and is replaced by an XJ6 ) , along the runway at Telluride Regional Airport , ( where May 's car is destroyed , and is replaced by an XJS ) and down the slopes of the Telluride Ski Resort . At Hanging Flume overlook , Clarkson experiences a traffic stop by a Mesa County police officer and reads the `` Her Britannic Majesty '' text from his British passport . Despite Clarkson 's car being the best , May admitting his car had poor brakes deemed the challenge pointless . The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street . Kiefer Sutherland beats Luke Evans to become `` the fastest actor with a middle name of George '' . 20 7 `` It 's a Gas , Gas , Gas '' Lamborghini Huracán Performante Anthony Joshua , Bill Goldberg 19 January 2018 ( 2018 - 01 - 19 ) The presenters discuss some new supercars and how handbrake turns impress women within the Conversation Street Segment . Hammond tests the Lamborghini Huracán Performante at the Eboladrome beating May 's Ferrari 458 Speciale in a drag race before Eaton takes it round the track setting a fast time . May and Hammond then look at ingenious ways of filling up with fuel instead of stopping at a fuel station . Anthony Joshua becomes `` the fastest man who makes a living out of punching or strangling someone '' beating Bill Goldberg , with both previously owning cars from General Motors . Clarkson then talks about the Rivalry in rallying between Audi with the Quattro and Lancia with the 037 . They conclude with a `` terrible disappointment '' that Lancia is no longer around . 21 8 `` Blasts from the Past '' Jaguar XK SS , Aston Martin DB4 GT , Honda Civic Type R , Ford GT Stewart Copeland , Nick Mason 26 January 2018 ( 2018 - 01 - 26 ) The presenters head to the City of Pau , Pyrénées - Atlantiques , France , and home of the Pau Grand Prix on the Circuit de Pau - Ville ( fr ) street course . At Palais Beaumont ( fr ) Hammond arrives with a Jagaur XK SS and Clarkson in an Aston Martin DB4 GT ; both being modern builds of older designs from the 1950s `` back - catalogues '' of Jaguar and Aston Martin . They both race around the Streets of Pau before May turns up in a modern Honda Civic Type R to prove that new cars are better . All three depart to drive south over the Pyrenees mountain range , with Clarkson and Hammond experiencing difficulties in dense fog . Clarkson recants a previous incident from 2013 -- when Clarkson and Hammond 's driving licences were confiscated whilst filming The Perfect Road Trip in France -- and tries to goad May into speeding on the Autopista AP - 2 . At Barcelona the presenters race around the abandoned Autódromo de Sitges - Terramar and the extreme banking on its corners , with May winning , and Hammond abstaining following his history of crashes . During Conversation Street the presenters discuss names for gritters . At the Eboladrome , Clarkson retests the Ford GT following the Niagara Falls race , with Eaton setting a test lap . Nick Mason goes on to record a faster lap than Stewart Copeland to become `` the fastest drummer for a band beginning with the letter p '' . 22 9 `` Breaking , Badly '' Jaguar XJ220 , Bugatti EB 110 Super Sport Dynamo , Penn & Teller 2 February 2018 ( 2018 - 02 - 02 ) The presenters discuss new sport cars by various companies , jumpers and the speed of pigeons vs blackbirds during Conversation Street . Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years by testing the Jaguar XJ220 and the Bugatti EB 110 Super Sport , the fastest supercars from the early 90s , both of which set slow times at the Eboladrome when driven by Eaton . The trio tries to break the UK speed record for amphibious vehicles by building one of their own . Clarkson oversees his idea of combining a Rolls - Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny . However on a nearby lake , Clarkson and May have to abort their run after achieving only 3 mph ( 5 km / h ) and nearly sinking . Hammond and May then go on to build a vehicle of their own by combining a Bond Bug with the engine of a personal water craft nicknaming it the `` Pond Bug '' . Competing on Coniston Water during Coniston Power Boat Records Week , Clarkson , being the only one with a speedboat licence , operates the `` Pond Bug '' and achieves an average of 47.81 mph ( 76.94 km / h ) , setting a new UK record . After performing a series of magic tricks , Dynamo and Penn & Teller compete to become `` the fastest magician in the world '' , with Penn & Teller prevailing on a snowy track . 23 10 `` Oh , Canada '' Alfa Romeo Stelvio Quadrifoglio , Range Rover Velar P380 , Porsche Macan Turbo Performance Pack , Tesla Model X , Ford F150 Raptor Rory McIlroy , Paris Hilton 9 February 2018 ( 2018 - 02 - 09 ) The presenters head to Canada to prove that small SUV 's are useless with Clarkson picking an Alfa Romeo Stelvio Quadrifoglio , May choosing a Range Rover Velar P380 and Hammond taking a Porsche Macan Turbo Performance Pack . They are shown to be slower round the Area 27 circuit than a BMW M3 , not practical enough to carry Canadian Newfoundland dogs , and slower round a barrel course than a horse : concluding that SUVs are pointless . Despite this , executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain . May abandons his Velar for his boat , which sinks , causing him to lose the race . Because Clarkson 's Stelvio has no tow bar he cheats and exchanges it for a Ford F150 Raptor , ultimately arriving first with the presenters concluding that the Raptor was better than all the SUVs being tested . On Conversation Street the presenters talk about new SUVs and vans . Clarkson reviews the Tesla Model X at the Eboladrome and on the road , accompanied by some lawyers , ostensibly to avoid Tesla pursuing any legal action , following the previous Tesla v. Top Gear court cases . Rory McIlroy and Paris Hilton discuss their car collections before McIlroy beats Hilton to become the `` World 's Fastest Golfing Enthusiast '' . 24 11 `` Feed the World '' None None 16 February 2018 ( 2018 - 02 - 16 ) The presenters head to Mozambique on a mission to feed the people of Bingo , an isolated town in the west of the country , with fish from the country 's capital of Maputo , covering a distance of approximately 200 miles . To do this , each of them attempt to find the best transport solution : May buys a Mercedes - Benz Estate and modifies it by fitting a Perspex tank filled with sea water in which to keep the fish alive ; Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine in an attempt to keep the fish frozen and Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike 's rear , inspired by a local method of drying and salting the fish to preserve them . After unsuccessfully trying to catch fish using local methods , they buy fish at a local market . Early in the journey , Clarkson 's truck crashes into May 's car upon rescuing it , damaging its radiator grille . Deeper into rural Mozambique , May constantly soaks himself with sea water from his ill - fitted tank and , due to his damaged radiator grille , suffers multiple breakdowns due to water intrusion . Hammond loses a large number of his fish on the bumpy roads , constantly gets stuck in thick mud and endlessly falls from his motorbike . Clarkson suffers very few issues except when his ice machine breaks , causing him to make small adjustments to his truck in order to smoke the fish with his exhaust . Upon May 's car breaking down in the middle of a lake , despite Hammond and Clarkson crossing successfully , Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck - much to May 's dismay and annoyance . Eventually , the three reach Bingo with May 's fish dead , Clarkson 's fish ruined and Hammond having almost no fish left . As a result , they sell no fish and leave Bingo in a helicopter `` like all charitable celebrities '' .
  Fact 0.150: Eaton sets a test lap.

Retrieved items:
<item 1>
The Grand Tour (TV series)
... we end the day at the Verkehrshaus , aka the Swiss Transport Museum , ... town of Kriens , 20 minutes from Lucerne , ... home to EVTech , ... We follow
Hammond as he plugs in his Rimac and then ... Waits ... why we end the day in the Kriens Chess Museum Jump up ^ Brown , Aaron ( 9 December 2017 ) . `` The Grand
Tour Season 2 : How to watch episode 1 , plus stream FULL second season for Free '' . Sunday Express . Retrieved 24 December 2017 . Dubbed `` Past , Present or
Future '' , this is the episode that features Hammond 's explosive car crash Jump up ^ The Grand Tour ( @ thegrandtour ) ( 5 December 2017 ) . `` 3 days to go
until # TheGrandTour returns to Prime Video with the episode Past , Present , or Future . www.PrimeVideo.com/GT '' ( Tweet ) -- via Twitter . Jump up ^ `` The
Grand Tour Season 2 '' . Amazon.co.uk website . ASIN B0731ZQ3R7. 1 .
</item 1>

<item 2>
The Grand Tour (TV series)
The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street . Kiefer Sutherland beats Luke Evans to become `` the fastest
actor with a middle name of George '' . 20 7 `` It 's a Gas , Gas , Gas '' Lamborghini Huracán Performante Anthony Joshua , Bill Goldberg 19 January 2018 ( 2018
- 01 - 19 ) The presenters discuss some new supercars and how handbrake turns impress women within the Conversation Street Segment . Hammond tests the
Lamborghini Huracán Performante at the Eboladrome beating May 's Ferrari 458 Speciale in a drag race before Eaton takes it round the track setting a fast time .
May and Hammond then look at ingenious ways of filling up with fuel instead of stopping at a fuel station . Anthony Joshua becomes `` the fastest man who makes
a living out of punching or strangling someone '' beating Bill Goldberg , with both previously owning cars from General Motors . Clarkson then talks about the
Rivalry in rallying between Audi with the Quattro and Lancia with the 037 .
</item 2>

<item 3>
The Grand Tour (TV series)
Rotten Tomatoes . Retrieved 17 July 2017 . Jump up ^ `` The Grand Tour season 2 episode 1 review : We 're in safe hands as Clarkson , Hammond and May return ''
. Digital Spy . Retrieved 8 December 2017 . Jump up ^ `` Television review : The Grand Tour on Amazon Prime '' . The Times . Retrieved 8 December 2017 . Jump up
^ `` The Grand Tour season 2 , episode 1 review : Has Jeremy Clarkson finally found his groove ? '' . The Daily Telegraph . Retrieved 8 December 2017 . Jump up
^ Clarkson , Jeremy ( 8 December 2017 ) . `` Jeremy Clarkson on Twitter '' . twitter.com . Retrieved 15 December 2017 . Jump up ^ Ling , Thomas ( 27 November
2017 ) . `` The Grand Tour 's first celebrity guests for series 2 have been revealed -- and there 's not a single woman '' . Retrieved 20 December 2017 . Jump
up ^ Long , Trevor .
</item 3>

<item 4>
The Grand Tour (TV series)
The Daily Telegraph , while not as positive , still approved of the episode , stating `` The writing is still rather ropey . Clarkson 's suggestion of a new
nickname for May -- `` Dingleberry Handpump '' -- failed to raise a titter even among the super-fans gathered for the London premiere '' but also said that ``
for each wobble , there are just as many moments when The Grand Tour manages the clever trick Top Gear could pull off at its best : raising a chuckle while
sneaking in a bit of serious journalism at the same time . '' and ultimately gave the episode 3 out of 5 stars . Jeremy Clarkson himself believed that they had
`` hit the ground running with series 2 of the Grand Tour '' . Radio Times has criticised the gender balance of guests on the `` Celebrity Face Off '' segment .
There was only 1 woman guest in the episode `` Oh Canada '' ; Paris Hilton . Hothouse Flowers playing for the opening sequence of `` The Holy Trinity '' episode
in the Lucerne Valley , California Broadcast ( edit ) Australian free - to - air network Seven Network started broadcasting the first series of The Grand Tour
in mid-October 2017 .
</item 4>

<item 5>
The Grand Tour (TV series)
The Sun . Retrieved 23 December 2017 . ^ Jump up to : Daly , Emma ( 26 November 2016 ) . `` Jeremy Clarkson forced to ' kill off ' celebs on The Grand Tour
after BBC ' bans Top Gear rip - off star interviews ' '' . The Sun . Retrieved 26 November 2016 . Jump up ^ Lee , Kristen ( 16 November 2017 ) . `` ' Celebrity
Brain Crash ' Axed on the Grand Tour Season 2 '' . jalopnik.com . Retrieved 17 November 2017 . Jump up ^ McCreesh , Louise ( 27 November 2017 ) . `` The Grand
Tour season 2 confirms celebrity guests including The Hoff , Jack Bauer and Luke Evans '' . Digital Spy . Retrieved 28 November 2017 . Jump up ^ Hsieh , Tony (
20 November 2016 ) . `` The Grand Tour : 3 Mustangs from the Opening Sequence of the Premiere Episode '' . Grand Tour Nation . Retrieved 24 November 2016 . ^
Jump up to : `` Amazon Instant Video '' .
</item 5>

<item 6>
The Grand Tour (TV series)
The Sunday Times : Driving . Retrieved 14 December 2017 . The show has a new test track in Enstone , ... there planning problems when ( Clarkson ) tried once
before to use the Enstone track , back in Top Gear days ... The noisier testing of fast cars will continue to be done in Wiltshire at RAF Wroughton ...
nicknamed `` the Eboladrome '' Jump up ^ Reed , Jason ( 24 January 2018 ) . `` The Secret Story Behind The Grand Tour 's Celebrity Braincrash '' . Shifting
Lanes . Retrieved 26 January 2018 . ^ Jump up to : McCreesh , Louise ( 27 November 2017 ) . `` The Grand Tour season 2 confirms celebrity guests including
Kiefer Sutherland and David Hasselhoff '' . Retrieved 19 December 2017 . The celebrity guests this series include Luke Evans , Kiefer Sutherland , Hugh
Bonneville , Kevin Pietersen and Dominic Cooper ( ... ) Dynamo , Rory Mcllroy , Michael Ball and Alfie Boe will also appear in season two .
</item 6>

<item 7>
The Grand Tour (TV series)
1 Dry 14 David Hasselhoff 1 : 24.1 Kevin Pietersen 1 : 17.2 Dry 15 Brian Wilson 1 : 17.5 Hugh Bonneville 1 : 22.2 Dry 16 Casey Anderson 1 : 18.6 Michael Ball 1
: 23.3 Damp 17 Alfie Boe 1 : 24.4 Dominic Cooper 1 : 23.6 Wet 18 Bill Bailey 1 : 25.1 Luke Evans 1 : 21.3 Dry 19 Kiefer Sutherland 1 : 17.8 Bill Goldberg 1 :
20.4 Dry 20 Anthony Joshua 1 : 18.7 Nick Mason 1 : 21.3 Dry 21 Stewart Copeland 1 : 24.2 Dynamo 1 : 39.3 Snow 22 Penn & Teller 1 : 33.8 Rory McIlroy 1 : 21.9
Wet 23 Paris Hilton 1 : 25.8 Conversation Street ( edit ) Introduced by Hammond using a car analogy to often take a left and park on the street , the trio
discuss car news .
</item 7>

<item 8>
The Grand Tour (TV series)
19 6 `` Jaaaaaaaags '' None Luke Evans , Kiefer Sutherland 12 January 2018 ( 2018 - 01 - 12 ) The presenters head to the area around Grand Junction , Colorado ,
staying at the Gateway Canyons Resort & Spa , and eventually travelling to Telluride , Colorado with a variety of old Jaguar cars to establish the manufacturer
's reputation for reliability with Clarkson buying an XJR , May taking an XK8 convertible and Hammond bringing a Mark X . During the course of the trip the cars
are driven around a dirt track ( where Hammond 's car is badly damaged , and is replaced by an XJ6 ) , along the runway at Telluride Regional Airport , ( where
May 's car is destroyed , and is replaced by an XJS ) and down the slopes of the Telluride Ski Resort . At Hanging Flume overlook , Clarkson experiences a
traffic stop by a Mesa County police officer and reads the `` Her Britannic Majesty '' text from his British passport . Despite Clarkson 's car being the best ,
May admitting his car had poor brakes deemed the challenge pointless .
</item 8>

<item 9>
The Grand Tour (TV series)
1 Wet not shown Volkswagen Up ! GTI 1 : 39.7 Wet 18 Enstone ( edit ) Gravel course at Enstone Airfield introduced in Series 2 and used for Celebrity Face Off
For the second series , a new test track was created at Enstone Airfield , close to the fixed studio tent location , and which the production team had
previously tried to use for Top Gear . The main Eboladrome track was planned to continue to serve for the testing of loud and fast cars . Celebrity Face off (
edit ) Following complaints regarding the `` Celebrity Brain Crash '' segment in the first series , this segment was replaced by `` Celebrity Face Off '' . Two
celebrities join the show each week , with a similar background or connection and often from different countries , to see which one of the two is faster . The
car used for this segment is a Jaguar F - Type R - Dynamic coupe , racing around a half gravel , half asphalt track at Enstone . Prior to showing the lap
results , Clarkson interviews the guests . Lap board Celebrity Time Condition Episode Ricky Wilson 1 : 20.1 Dry 14 David Hasselhoff 1 : 24.
</item 9>

<item 10>
The Grand Tour (TV series)
ASIN B0731ZQ3R7. 1 . Past , present or future Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : Starkey , Adam ( 13
December 2017 ) . `` The Grand Tour season 2 : Richard Hammond is on crutches in episode 2 after horrifying crash '' . Metro . London . Retrieved 13 December
2017 . second episode titled The Falls Guys , Clarkson drives ... Ford GT from New York City to Niagara Falls , with James May making the same journey via
public transport with Hammond ... Celebrity Face Off ... cricketer Kevin Pietersen and baseball 's Brian Wilson Jump up ^ `` The Grand Tour Season 2 Episode 3
'' . 16 December 2017 . Retrieved 16 December 2017 . ^ Jump up to : `` Zadarska manekenka Silvija oduševila u ' The Grand Touru ' '' ( Zadar model Silvija
delights in The Grand Tour ) . 24sata ( in Croatian ) . 3 January 2018 .
</item 10>
=== END ITEM id=a176-d2f6521da5f57b8c ===

=== ITEM id=a189-7cfa29d2bd815178 ===
Gold sentences and facts:
Sentence 0: State Pre-Prohibition ( prior to 1919 ) Post-Prohibition ( after 1933 ) 1970s / 26th Amendment ( adopted in 1971 ) 1980s / Drinking Age Act of 1984 21st century Alabama Pre 1881 : None 1881 : 21 ( none with consent of parent ) 21 1975 : Lowered to 19 1986 : Raised to 21 21 ( no one underage is allowed consumption Section 28 - 1 ) Alaska ? 21 1970 : Lowered to 19 1984 : Raised to 21 with grandfather clause ( if born before January 1 , 1965 ) 21 Arizona ? 21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and liquor 18 Raised to 21 in 1987 21 Connecticut ? 21 1972 : Lowered to 18 1982 : Raised to 19 1983 : Raised to 20 1985 : Raised to 21 21 ( exemptions : ( 1 ) a person over age eighteen who is an employee or permit holder under section 30 - 90a and who possesses alcoholic liquor in the course of such person 's employment or business , ( 2 ) a minor who possesses alcoholic liquor on the order of a practicing physician , or ( 3 ) a minor who possesses alcoholic liquor while accompanied by a parent , guardian or spouse of the minor , who has attained the age of twenty - one . Nothing in this subsection shall be construed to burden a person 's exercise of religion under section 3 of article first of the Constitution of the state in violation of subsection ( a ) of section 52 - 571b . Delaware ? 21 1972 : Lowered to 20 1984 : Raised to 21 21 District of Columbia ? 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1986 : Raised to 21 with grandfather clause . 21 Florida ? 18 1980 : 19 1985 : Raised to 21 with grandfather clause 21 Georgia ? 21 1972 : Lowered to 18 1984 : Raised to 19 1985 : Raised to 20 1986 : Raised to 21 21 Hawaii ? 21 1972 : Lowered to 18 1986 : Raised to 21 21 Idaho ? 20 : for beer 21 : for wine and liquor 1972 : Lowered to 19 ( all ) 1987 : Raised to 21 in 1987 ( Apr 11 ) with grandfather clause 21 Illinois 1872 : Age of majority ( none with written consent of parent ) pre 1961 : Age of majority 1961 : Raised to 21 1973 : Lowered to 19 : for beer and wine 21 : for liquor 1980 : Raised to 21 21 Indiana ? Post-1934 : 21 21 21 21 Iowa ? 21 1972 : Lowered to 19 in 1972 ( Jul 1 ) 1973 : Lowered to 18 in 1973 ( Jul 1 ) 1978 : Raised to 19 in 1978 ( Jul 1 ) 1986 : Raised to 21 in 1986 ( Jul 1 ) with grandfather clause . 21 Kansas ? 1949 : 18 : for beer and wine with less than 4 % ABV 21 : for liquor 18 : for beer and wine with less than 4 % ABV 21 : for beer and wine with 4 % ABV or more 21 : for liquor 1985 : Raised to 21 21 ( However , anyone under 21 may consume cereal malt beverages with parental supervision on their own property ) Kentucky ? 1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 : De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision , raising to 21 three months later . Other exceptions still remain . 21 ( Exceptions to state law include : For an established religious purpose ; When a person under twenty - one years of age is accompanied by a parent , spouse , or legal guardian twenty - one years of age or older ; For medical purposes when purchased as an over the counter medication , or when prescribed or administered by a licensed physician , pharmacist , dentist , nurse , hospital , or medical institution ; In a private residence , which shall include a residential dwelling and up to twenty contiguous acres , on which the dwelling is located , owned by the same person who owns the dwelling ; The sale , handling , transport , or service in dispensing of any alcoholic beverage pursuant to lawful ownership of an establishment or to lawful employment of a person under twenty - one years of age by a duly licensed manufacturer , wholesaler , or retailer of beverage alcohol . ) Maine ? 21 1969 : Lowered to 20 1972 : Lowered to 18 1977 : Raised to 20 1985 : Raised to 21 21 Maryland ? 21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses , teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18 in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21 Mississippi ? 1966 : 18 : for beer and wine 21 : for liquor ( alcohol not legalized until 1966 ) 18 Raised to 21 in 1986 21 Missouri ? 1945 : 21 21 21 21 Montana ? 21 1971 : Lowered to 19 1973 : Lowered to 18 1979 : Raised to 19 1987 : Raised to 21 21 Nebraska ? 21 1969 : Lowered to 20 1972 : Lowered to 19 1980 : Raised to 20 1985 : Raised to 21 21 Nevada ? 1933 : 21 21 21 21 New Hampshire ? 21 1973 : Lowered to 18 1979 : Raised to 20 1985 : Raised to 21 21 New Jersey Before 1880 : None Post-1880 : 18 ( penalties only against businesses ) 21 1973 : Lowered to 18 1980 : Raised to 19 1983 : Raised to 21 21 New Mexico ? 1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 : for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ? 1935 : 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : beer 21 : wine and liquor 1988 : 21 21 ( However , anyone can drink with parents or legal - age spouse ) Oklahoma ? Men : 21 Women : 18 : 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1976 : Lowered to 18 ( for 3.2 % ABW beer ) for both sexes in 1976 ( Dec ) Last state to lower the drinking age . 1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to 20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 21 South Dakota ? 1934 : 21 ( all ) 1939 : Lowered to 18 : for 3.2 % beer Raised to 19 : for 3.2 % beer 1972 : 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine 21 : for liquor 1988 : Raised to 21 21 Tennessee Pre 1881 : None 1881 : 21 21 1964 : Lowered to 18 1979 : Raised to 19 1984 : Raised to 21 21 Texas Pre 1909 : 21 None ( with written permission of parent or guardian ) Post 1909 : 21 21 1973 : Lowered to 18 1981 : Raised to 19 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 Utah ? 1935 : 21 21 21 21 Vermont ? 21 1971 : Lowered to 18 1986 : Raised to 21 21 Virginia ? 21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer ) 1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all ) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all ) 18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales , remained 18 for on - premises ) . 1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 ( however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010
  Fact 0.86: In 1996 the Louisiana Supreme Court briefly lowered the drinking age to 18, until it reversed its decision and raised it to 21 three months later.

Retrieved items:
<item 1>
U.S. history of alcohol minimum purchase age by state
Minimum legal drinking age as of 1983 ( one year before the Drinking Age Act of 1984 was passed ) : Minimum age is 21 Minimum age is 20 Minimum age is 19 and 21
Minimum age is 19 Minimum age is 18 and 21 Minimum age is 18 The alcohol laws of the United States regarding minimum age for purchase have changed over time .
The history is given in the table below . Unless otherwise noted , if different alcohol categories have different minimum purchase ages , the age listed below
is set at the lowest age given ( e.g. if the purchase age is 18 for beer and 21 for wine or spirits , as was the case in several states , the age in the table
will read as `` 18 '' , not `` 21 '' ) . In addition , the purchase age is not necessarily the same as the minimum age for consumption of alcoholic beverages ,
although they have often been the same . As one can see in the table below , there has been much volatility in the states ' drinking ages since the repeal of
Prohibition in 1933 .
</item 1>

<item 2>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer )
1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all
) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all )
18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales ,
remained 18 for on - premises ) .
</item 2>

<item 3>
U.S. history of alcohol minimum purchase age by state
1938 : 21 21 21 21 Louisiana ? As of 1948 : 18 18 1987 : Raised to 21 de jure in 1987 , but de facto age was still 18 until 1995 due to a sale loophole . 1995 :
De facto age raised to 21 in 1995 when loophole was closed . 1996 : In 1996 , briefly lowered by Louisiana Supreme Court to 18 until it reversed its decision ,
raising to 21 three months later . Other exceptions still remain .
</item 3>

<item 4>
U.S. history of alcohol minimum purchase age by state
U.S. history of alcohol minimum purchase age by State - wikipedia U.S. history of alcohol minimum purchase age by State Jump to : navigation , search Minimum
legal drinking age as of 1969 ( two years before the 26th Amendment was passed ) : Minimum age is 21 Minimum age is 20 Minimum age is 19 and 21 Minimum age is
19 Minimum age is 18 and 21 Minimum age is 18 Minimum legal drinking age as of 1975 ( when almost all states had their lowest age limit ) : Minimum age is 21
Minimum age is 20 Minimum age is 19 and 21 ; see below Minimum age is 19 Minimum age is 18 and 21 ; see below Minimum age is 18 Detail on dual age limits Both
age limits apply for following states : Washington , D.C. : The legal drinking age is 18 for beer and wine , and 21 for liquor . Illinois : The legal drinking
age is 19 for beer and wine , and 21 for liquor .
</item 4>

<item 5>
U.S. history of alcohol minimum purchase age by state
Shortly after the ratification of the 21st amendment in December , most states set their purchase ages at 21 since that was the Voting age at the time . Most of
these limits remained constant until the early 1970s . From 1969 to 1976 , some 30 states lowered their purchase ages , generally to 18 . This was primarily
because the voting age was lowered from 21 to 18 in 1971 with the 26th amendment . A lot of states started to lower their minimum drinking age in response ,
most of this occurring in 1972 or 1973 . Twelve states kept their purchase ages at 21 since repeal of Prohibition and never changed them . From 1976 to 1983 ,
several states voluntarily raised their purchase ages to 19 ( or , less commonly , 20 or 21 ) , in part to combat drunk driving fatalities . In 1984 , Congress
passed the National Minimum Drinking Age Act , which required states to raise their ages for purchase and public possession to 21 by October 1986 or lose 10 %
of their federal highway funds .
</item 5>

<item 6>
U.S. history of alcohol minimum purchase age by state
1972 : Lowered to 18 ( all ) in 1972 ( Mar ) 1977 : Border state restriction lifted . 1984 : Raised to 19 1986 : Raised to 21 ( September 1 , 1986 ) 21 (
however anyone can drink when parents , spouses , teachers are present ) Wyoming ? 21 1973 : Lowered to 19 1988 : Raised to 21 in 1988 ( Jul 1 ) Last state to
raise de jure drinking age to 21 21 American Samoa ? ? ? 21 ? 21 Northern Mariana Islands -- -- ? 21 ? 21 Puerto Rico ? 18 18 Kept at 18 despite 10 % highway
funding penalty under Drinking Age Act . 18 Virgin Islands ? ? ? 18 18 Guam ? 21 ? 18 18 Raised to 21 in 2010 Additional Notes ( edit ) Contrary to popular
belief , since the National Minimum Drinking Age Act of 1984 , not all states specifically prohibit minors ' and young adults ' consumption of alcohol in
private settings .
</item 6>

<item 7>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine 21 : for liquor 1982 : Raised to 21 with grandfather clause 21 ( however anyone can drink when parents , spouses ,
teachers are present ) Massachusetts ? 21 1973 : Lowered to 18 1979 : Raised to 20 in 1979 ( Apr 16 ) Raised to 21 in 1985 21 Michigan ? 21 1972 : Lowered to 18
in 1972 ( Jan 1 ) 1978 : Raised to 19 in 1978 ( Dec 3 ) 1978 : Raised to 21 in 1978 ( Dec 21 ) , 18 days later . First state to raise age to 21 since it was
lowered 21 Minnesota ? 21 1973 : Lowered to 18 ( June 1 ) 1976 : Raised to 19 ( Sept 1 ) First state to raise age after lowering . 1986 : Raised to 21 21
Mississippi ?
</item 7>

<item 8>
U.S. history of alcohol minimum purchase age by state
21 1972 : Lowered to 19 1985 : Raised to 21 21 Arkansas Pre 1925 : None 1925 : 21 21 21 21 21 California Pre 1891 : Regulated by municipality / county ( common
age was 16 ) 1891 : 18 ( statewide ) 1933 : 21 21 21 21 ( except small amounts for religious ceremonies ) Colorado None 1945 : 18 : for beer 21 : for wine and
liquor 18 Raised to 21 in 1987 21 Connecticut ?
</item 8>

<item 9>
U.S. history of alcohol minimum purchase age by state
By mid-1988 , all 50 states and the District of Columbia had raised their purchase ages to 21 ( but not Puerto Rico , Guam , or the Virgin Islands , see
Additional Notes below ) . South Dakota and Wyoming were the final two states to comply with the age 21 mandate . The current drinking age of 21 remains a point
of contention among many Americans , because of it being higher than the age of majority ( 18 in most states ) and higher than the drinking ages of most other
countries . The National Minimum Drinking Age Act is also seen as a congressional sidestep of the tenth amendment . Although debates have not been highly
publicized , a few states have proposed legislation to lower their drinking age , while Guam has raised its drinking age to 21 in July 2010 .
</item 9>

<item 10>
U.S. history of alcohol minimum purchase age by state
1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 :
for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ?
</item 10>
=== END ITEM id=a189-7cfa29d2bd815178 ===

=== ITEM id=a148-eace0b33d72b37ae ===
Gold sentences and facts:
Sentence 0: No . Official English title Original Japanese title Original air date English air date 151 `` Sabertooth '' `` Seibātūsu '' ( 剣 咬 の 虎 ( セイバー トゥース ) ) October 6 , 2012 November 17 , 2015 The Legion Platoon is tasked by Zentopia 's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again . Before embarking on their mission , they visit Fairy Tail and make amends with the guild for their previous hostility against them . It is later discovered that Fairy Tail has lost their title as the strongest guild in Fiore to the Saber Tooth guild over the past seven years . Two of the strongest wizards in the guild include a pair of Dragon Slayers named Sting and Rogue , who are obsessed with fighting Natsu and Gajeel , respectively . Meanwhile , Makarov brings Gildarts to their abandoned guildhall and shows him Lumen Histoire , a light that Makarov says is Fairy Tail 's greatest secret . Makarov then tells Gildarts that he has been chosen to become the next master of Fairy Tail . 152 `` And So We Aim for the Top '' `` Soshite Ore - tachi wa Chōjō o Mezasu '' ( そして オレ たち は 顶 上 を 目指す ) October 13 , 2012 November 24 , 2015 Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island . The group meets with Poluchka for medicine to strengthen their magic , but are unsuccessful . Wendy recognizes Poluchka 's voice and scent as those of Grandine , the dragon who raised her . Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her . Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 `` Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction . When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years . They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild , Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power . The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave . 155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray , Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus . As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' `` Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first , only to discover they have barely taken last place . Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace , Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla . To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced , they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars '' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once . Humiliated , Gray swears revenge against Raven Tail and Saber Tooth . Afterwards , Lucy is chosen to fight Raven Tail member Flare in the tournament 's first battle round . 159 `` Lucy vs. Flare '' `` Rūshii vs. Furea '' ( ルーシィ vs. フレア ) December 1 , 2012 January 12 , 2016 Lucy displays tremendous skill in her battle against Flare by being able to summon two spirits simultaneously . As Lucy gains the upper hand , Flare cheats by using her magic hair to hold Alzack and Bisca 's daughter Asuka as a hostage , blackmailing Lucy into not fighting back to keep her from harming Asuka . As Lucy tearfully speaks of Asuka 's danger , Natsu is able to hear it and rushes to Asuka to incinerate Flare 's hair , allowing Lucy to counterattack with her Urano Metoria spell in combination with Gemini . However , before the spell fully activates , Flare 's teammate Obra nullifies Lucy 's magic , causing her to collapse and lose the match . 160 `` Portent '' `` Kyōzui '' ( 凶 瑞 ) December 8 , 2012 January 19 , 2016 Blue Pegasus ' Ren defeats Mermaid Heel 's Araña during the games ' second match , while Saber Tooth 's Orga effortlessly defeats Quatro Cerberus ' War Cry in the third match . For the day 's fourth and final match , Jellal is pitted against Lamia Scale 's Jura , intending to avenge the humiliated Fairy Tail members . He mimics Mystogan 's magic to keep up his disguise , but he finds it ineffective to use against Jura and is forced to use his own magic instead . Before he can cast a spell that will expose his true identity , Ultear uses Merudy 's sensory to force Jellal into submission . Meanwhile , the hospitalized Carla awakens after having a troubling premonition of the future . 161 `` Chariots '' `` Chariotto '' ( 战 车 ( チャリオット ) ) December 15 , 2012 January 26 , 2016 Fairy Tail is undaunted by their poor performance on the first day . As the guild celebrates , they are visited and taunted by Bacchus , Erza 's rival from Quatro Cerberus , who has come to replace War Cry . Carla relays her premonition to Poluchka , saying that she saw Mercurius collapsing around what appeared to be a tearfully singing Lucy . The second day of the tournament begins with a footrace atop a train of moving chariots , causing the participants Natsu , Gajeel , and Sting to suffer from motion sickness as a side effect of their Dragon Slayer magic . As Natsu and Gajeel force themselves forward , Sting inquires about their determination to which Natsu responds that it is as a result of the strength of Fairy Tail 's bonds . Sting eventually throws the race while Natsu and Gajeel manage to finish as sixth and seventh placers , respectively , earning Fairy Tail the audience 's respect . 162 `` Elfman vs. Bacchus '' `` Erufuman vs. Bakkasu '' ( エルフマン vs. バッカス ) December 22 , 2012 September 1 , 2015 Due to a miscomunnication between the king and his knight Arcadios , Elfman is chosen as Bacchus ' opponent instead of Erza . Before the match commences , the two opponents make a wager : if Bacchus wins , he will court Elfman 's sisters , and if Elfman wins , Bacchus ' team name will be changed to `` Quatro Puppy '' for the rest of the games . Bacchus ruthlessly attacks Elfman with his bare palms , who manages to damage Bacchus with his defensive Lizardman transformation . Elfman manages to outlast Bacchus as he exhausts himself , winning the match and forcing Bacchus ' team to be renamed . Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his `` Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts . Jenny then transforms into her battle armor while Mirajane shifts into her Satan Soul `` Sitri '' form and easily defeats Jenny . The final match of the day is announced , with Kagura Mikazuchi of Mermaid Heel pitted against Yukino Aguria of Saber Tooth . Yukino proposes to Kagura that they wager their lives on their battle . Meanwhile , Arcadios is confronted by Darton , the kingdom 's defense minister , about his `` Eclipse Project '' . 164 `` Kagura vs. Yukino '' `` Kagura vs. Yukino '' ( カグラ vs. ユキノ ) January 12 , 2013 September 1 , 2015 Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura . Using her gravity - altering magic , Kagura defeats the two spirits , thus forcing Yukino to use a Gatekey that summons Ophiuchus , which is regarded as the strongest among the Celestial Spirits . However , Kagura slays Ophiuchus with her sword still sheathed and easily defeats Yukino . Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance . Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete '' ( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ; however , Lucy turns down the offer . Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal . Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength . The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points . Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent scores depending on the extent of power used . In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei '' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve . For the third match , Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail . Since the opponent is from a suspicious guild , the Fairy Tail members take extra precautions by watching the actions of Ivan Dreyar as well as the other four Raven Tail members . However , as the match commences , the spectators are shocked to see Laxus being beaten by Alexei , although no cautious movements can be seen from the Raven Tail team . It turns out that the two dueling people are just illusions created by Alexei in order for him to have a talk with Laxus . Also , the other Raven Tail members being monitored by Fairy Tail are just lifelike illusions , for in reality they are with Alexei in the battlefield . Alexei reveals himself to be Ivan Dreyar . Unable to interrogate Laxus of the Lumen Histoire 's location , Ivan and the Raven Tail members are forced to fight him . Before their battle begins , Ivan declares that Raven Tail is an anti-Fairy Tail guild . 169 `` Wendy vs. Shelia '' `` Wendi vs. Sheria '' ( ウェンディ vs. シェリア ) February 16 , 2013 September 1 , 2015 After Laxus defeats Ivan and the four Raven Tail elite members , Ivan 's illusions disappear to the shock of everyone in the stadium . The game officials declare Laxus the victor , and Ivan tells him that Lumen Histoire is Fairy Tail 's ' darkness ' . Due to the infraction of the rules , the Raven Tail team is disqualified for the remainder of the Games , and its members are captured by the soldiers . Later , Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off to compete for the final match of the day . As the two exchange attacks , gusts of wind surround the arena since Sherria is revealed to be a Sky God Slayer . Wendy employs an advanced Sky Dragon Slayer attack much to Poluchka 's astonishment , causing Sherria to fall on the ground . Wendy is about to be announced as the winner of the bout when Sherria suddenly stands up with her wounds fully healed . Meanwhile , the Crime Sorcière guild senses the nefarious aura , and Jellal proceeds to the Domus Flau audience , oblivious that he bumped into Doranbolt , who , in turn , wonders what is Mystogan doing in Earthland . 170 `` Small Fists '' `` Chiisana Kobushi '' ( 小さな 拳 ) February 23 , 2013 September 1 , 2015 With Cheria being fully healed and Wendy being almost depleted of magic power , their battle continues , with the former casting a high - level spell towards the latter . However , it seems that Wendy enhances Cheria 's physical strength , causing the attack to gain much force and thus missing Wendy . The two competitors then engage in a hand - to - hand combat , but due to the time limit and neither side is giving up , their battle resulted in a draw , giving both teams with 5 points . Meanwhile , Jellal manages to track the sinister aura among the crowd in the form of a mysterious , hooded person . Jellal is about to confront the aura source , but he is abruptly cornered by Doranbolt , Lahar , and other Rune Knights . Doranbalt then interrogates Jellal , and as his cover was blown , the Rune Knights , as well as Kagura and Milliana glancing nearby , are startled to see the true identity of Mystogan . Yajima convinces the Rune Knights that Mystogan is Jellal 's Edolas counterpart , prompting Lahar to release the fugitive . Meanwhile , Laxus and Makarov have a small conversation regarding Lumen Histoire when Mavis appears and explains that Lumen Histoire is the ' light ' of Fairy Tail . 171 `` Naval Battle '' `` Nabaru Batoru '' ( 海战 ( ナバルバトル ) ) March 2 , 2013 September 1 , 2015 As the jubilant Fairy Tail members celebrate their victories , the other people , such as Sting , Kagura , and Milliana , are contemplating on their own concerns , while Yukino is enlisted as a sergeant under Arcadios ' command . Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others rush to her aid . 172 `` A Parfum for You '' `` Kimi ni Sasageru Parufamu '' ( 君 に 捧げる 香り ( パルファム ) ) March 9 , 2013 September 1 , 2015 While Lucy is hospitalized and taken to the infirmary , the Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so . Later , Makarov announces that the two Fairy Tail teams must merge as a result of the disqualification of Raven Tail and to compensate for the odd number of teams for the battle round . He also reveals that the new team will have the former Team A 's 35 points to start with . The new Fairy Tail team , consisting of Natsu , Gray , Erza , Gajeel , and Laxus , are well - received as they enter the field . The second part of the tournament begins with a tag battle between Blue Pegasus ' Ichiya and `` Rabbit '' and Quatro Puppy 's Rocker and Bacchus . To everyone 's shock , it is revealed that `` Rabbit '' is actually Nichiya in disguise , and after the Exceed is punched by Bacchus , Ichiya bulks up to defeat the Quatro Puppy opponents , awarding Blue Pegasus with 10 points . 173 `` Battle of Dragon Slayers ! '' `` Batoru obu Doragon Sureiyā '' ( バトル ・ オブ ・ ドラゴン スレイヤー ) March 16 , 2013 September 1 , 2015 Mermaid Heel 's Kagura and Milliana and Lamia Scale 's Lyon and Yuka face off for the second tag battle of the games . The two teams struggle to earn a victory , only to have their battle ending in a draw due to the time limit . For the third and final match of the day , the spectators look on in silence as Natsu and Gajeel of Team Fairy Tail and Sting and Rogue of Team Saber Tooth enter the field . As the battle commences , Sting and Rogue were about to take the initiative when they were suddenly attacked by Natsu and Gajeel . The four mages then exchange attacks ; however , the audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers . Unable to keep themselves on par with their adversaries , Sting activates ' White Drive ' while Rogue powers up using ' Shadow Drive ' . 174 `` Four Dragons '' `` Yonin no Doragon '' ( 四 人 の 竜 ( ドラゴン ) ) March 23 , 2013 September 1 , 2015 The tag battle continues , with Sting and Rogue beginning to attack Natsu and Gajeel with their powers being increased . After receiving heavy blows from the Saber Tooth Dragon Slayers , Natsu and Gajeel manages to hold them off , thus forcing Sting and Rogue to activate `` Dragon Force '' on their own will , to the shock of the Fairy Tail audience . Sting decides to finish the Fairy Tail Dragon Slayers alone , for he held a promise to his Exceed friend Lector many years ago . As the three mages fight , the tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel , resulting in the collapse of the arena floor , and after a hand - to - hand combat underground , Sting was seemingly the victor of the match . However , the Fairy Tail duo stands up , comments on Sting 's strength , and argues about his position while attacking , thereby causing Natsu to drag Gajeel on a mining cart and send it away . While Gajeel is left incapacitated due to motion sickness , Natsu challenges the Twin Dragons of Saber Tooth alone , in retribution to what Sting did earlier on them . 175 `` Natsu vs. the Twin Dragons '' `` Natsu vs. Sōryū '' ( ナツ vs. 双竜 ) March 30 , 2013 September 1 , 2015 The tag battle of the Dragon Slayers comes to a close , with Natsu overpowering the combined attacks of Sting and Rogue . The Saber Tooth members watch in dismay , while Jiemma seethes in anger over the defeat of the Twin Dragons , frightening the nearby spectators away . The rest of Fairy Tail celebrates in joy as its team receives 10 points , finally surpassing Team Saber Tooth by a single point . Meanwhile , Gajeel , back to his senses after being dragged by Natsu into a cart earlier , stumbles upon a cave full of dragon skeletons . During that night , Jellal manages to corner the source of the dark aura , standing in disbelief at the identity of the mysterious , hooded person .
  Fact 0.308: Episode 173 aired in English on September 1, 2015.

Retrieved items:
<item 1>
List of Fairy Tail episodes (season 6)
The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled
with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests
the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters
using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points .
Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes
abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent
scores depending on the extent of power used .
</item 1>

<item 2>
List of Fairy Tail episodes (season 6)
Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her
. Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of
resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest
guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 ``
Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu
and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of
destruction .
</item 2>

<item 3>
List of Fairy Tail episodes (season 6)
It contains a single story arc , titled `` Dai Matō Enbu '' ( 大 魔 闘 演武 , lit `` Grand Magic Games '' ) , which adapts material from the beginning of the 31st
through the ending of the 35th volumes of Hiro Mashima 's Fairy Tail manga over 24 episodes . Focusing on Natsu and the others who were frozen in time for seven
years on Tenrou Island , the members return to Magnolia and discover that Saber Tooth is the strongest guild in the Fiore Kingdom . Having become the weakest
guild during the absence of its core members , Fairy Tail decides to enter the Grand Magic Games , an annual competition to decide Fiore 's strongest guild .
The season began airing on October 6 , 2012 on TV Tokyo in Japan and concluded its run on March 30 , 2013 . The first DVD compilation , containing the last two
episodes of the fifth season and the first two episodes of the sixth season , was released by Pony Canyon on March 6 , 2013 .
</item 3>

<item 4>
List of Fairy Tail episodes (season 6)
To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as
Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced ,
they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars
'' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to
score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who
scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once .
</item 4>

<item 5>
List of Fairy Tail episodes (season 6)
Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy
by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his ``
Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand
Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for
being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes
redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts .
</item 5>

<item 6>
List of Fairy Tail episodes (season 6)
Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace ,
Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is
weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy
Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail
guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is
Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla .
</item 6>

<item 7>
List of Fairy Tail episodes (season 6)
The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic
Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the
painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza
can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave .
155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray
, Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus .
</item 7>

<item 8>
List of Fairy Tail episodes (season 6)
In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's
victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the
two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked
leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei
'' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of
Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve .
</item 8>

<item 9>
List of Fairy Tail episodes (season 6)
Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza
reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance
. Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete ''
( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete
the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ;
however , Lucy turns down the offer .
</item 9>

<item 10>
List of Fairy Tail episodes (season 6)
Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water
sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva
ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to
surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes
attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others
rush to her aid .
</item 10>
=== END ITEM id=a148-eace0b33d72b37ae ===
