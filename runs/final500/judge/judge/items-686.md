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

=== ITEM id=34f24f38a90aa7fa ===
Gold sentences and facts:
Sentence 0: In episode 3 , `` Hugs Can Be Deceiving '' , Suzanne 's backstory is revealed via a flashback . She was adopted by white parents and raised in the suburbs , but her mental instability alienated her classmates and their parents alike . Her adoptive mother pushed her to excel at everything she did to prove that she was as good as everyone else ; ironically , the constant pressure only made Suzanne 's psychological problems worse . In the same episode , it is revealed that Suzanne had run outside and knocked Chapman unconscious after Chapman had beaten Tiffany `` Pennsatucky '' Doggett ( Taryn Manning ) unconscious at the end of season 1 .
  Fact 0.0: In episode 3, titled 'Hugs Can Be Deceiving', Suzanne's backstory is revealed via a flashback.
  Fact 0.1: Suzanne was adopted by white parents.
  Fact 0.2: Suzanne was raised in the suburbs.
  Fact 0.3: Suzanne's mental instability alienated her classmates.
  Fact 0.4: Suzanne's mental instability alienated her classmates' parents.
  Fact 0.5: Suzanne's adoptive mother pushed her to excel at everything she did, to prove she was as good as everyone else.
  Fact 0.6: Ironically, the constant pressure from Suzanne's adoptive mother made Suzanne's psychological problems worse.
  Fact 0.7: In the same episode, it is revealed that Suzanne had run outside.
  Fact 0.8: In the same episode, Suzanne knocked Chapman unconscious.
  Fact 0.9: Suzanne knocked Chapman unconscious after Chapman had beaten Tiffany 'Pennsatucky' Doggett (Taryn Manning) unconscious.
  Fact 0.10: Chapman beat Doggett unconscious at the end of season 1.

Retrieved items:
<item 1>
Crazy Eyes (character)
Crazy Eyes ( character ) - wikipedia Crazy Eyes ( character ) Jump to : navigation , search Suzanne `` Crazy Eyes '' Warren Orange Is the New Black character
Uzo Aduba as Suzanne `` Crazy Eyes '' Warren in a promotional poster for season 2 of Orange Is the New Black First appearance `` I Was n't Ready '' Season 1 ,
episode 1 July 11 , 2013 Portrayed by Uzo Aduba Eden Wiggins ( at age 10 ) Taliyah Whitaker ( at age 5 ) Information Significant other ( s ) Maureen Kukudio (
ex-girlfriend ) Parents Pat and Dennis Warren Suzanne `` Crazy Eyes '' Warren is a fictional character played by Uzo Aduba on the Netflix series Orange Is the
New Black . Warren is portrayed as intelligent , but lacking in social skills , and prone to spiral into emotional outbursts when agitated . The character is
the only role that has received Emmy Award recognition both in the comedy and drama genres from the same show and only the second character to earn Emmy
recognition in both genres .
</item 1>

<item 2>
Graph query row:
entity: Suzanne Warren
relationship: RELATED_TO
detail: character on
related: Orange Is the New Black
</item 2>

<item 3>
Graph query row:
entity: Suzanne Warren
relationship: RELATED_TO
detail: child of
related: Dennis Warren
</item 3>

<item 4>
Graph query row:
entity: Suzanne Warren
relationship: RELATED_TO
detail: child of
related: Pat
</item 4>

<item 5>
Crazy Eyes (character)
Suzanne nicknames Chapman `` Dandelion '' , because of her blonde hair and puts in a request to become Chapman 's bunkmate . Nonetheless , Chapman resists her
advances . As a result , Suzanne urinates on the floor in Chapman 's bunk quarters . In episode 9 , `` Fucksgiving '' , her white adoptive parents , Pat and
Dennis Warren , are introduced . Season 2 ( edit ) As she rebounds from her unrequited love for Chapman , Suzanne is befriended by veteran inmate Yvonne `` Vee
'' Parker ( Lorraine Toussaint ) as Vee develops her powerbase in the prison . Vee takes advantage of Suzanne 's desperation for affection , and Suzanne becomes
Vee 's fanatically loyal right - hand woman . Suzanne serves Vee as muscle , even beating up her friend Poussey Washington ( Samira Wiley ) on Vee 's orders .
When Vee attacks prison matriarch Galina `` Red '' Reznikov ( Kate Mulgrew ) and puts her in the hospital ward , she manipulates Suzanne into believing that she
committed the crime .
</item 5>

<item 6>
Crazy Eyes (character)
Eventually , she turns to new prison counsellor Berdie Rogers 's drama class . There , she begins writing a surrealistic science - fiction erotica series called
`` The Time Hump Chronicles '' which , although considered obscene and strange by both Rogers and Taystee , becomes an instant hit amongst the inmates .
Although they prove to be therapeutic for her mental health , Suzanne becomes stressed and irritated with the constant harassment by the other inmates for more
material , the persistent ideas that are being floated past her and the fan - fiction that is left at her bunk or lying around . Soon , some of the extracts
find their way into the hands of the COs , resulting in Officer Donaldson being mocked by both his colleagues and the inmates alike upon the realization that he
is the inspiration for one of the characters . This leads to Rogers ' suspension . Meanwhile , Suzanne is taken by surprise when she discovers that one of her
more prolific fans , Maureen Kukudio , is actually interested in her romantically . Suzanne seeks advice from Lorna about how to respond ; she admits she finds
Maureen attractive , but she has never had a girlfriend before and subsequently feels she must back away from a sexual liaison .
</item 6>

<item 7>
Crazy Eyes (character)
It is mentioned that she took a plea deal . When the inmates take advantage of a temporary lack of security and enjoy the nearby lake and woods , Kukudio
insists that she and Suzanne run away together . Suzanne refuses and their relationship becomes rocky . In `` People Persons , CO Humphrey commands the two of
them to fight . Suzanne rejects this but Maureen , who wants to fight , provokes her by launching into a stream of insults . Suzanne fully retaliates by
throwing Maureen to the ground and beating her repeatedly , until the Dominicans pull her off . This fight leads her to extreme guilt and a wish to punish
herself . She demonstrates this especially when a peaceful protest in the jail cafeteria turns into a chaotic scene and Poussey Washington dies by suffocation .
Whilst the inmates grieve over her death , Suzanne believes that Poussey is alive and tries numerous times to suffocate herself . Finally , she is sent to
Medical and is placed in a bed next to Kukudio . Season 5 ( edit ) In season five , Suzanne continues to recover from her injuries and also goes out of her way
to help Maureen , whose injuries are severe .
</item 7>

<item 8>
Crazy Eyes (character)
In the season finale , Suzanne assists Poussey and Taystee in caring for Brook Soso , whom they discover unconscious from a drug overdose , and develops a
closer bond with Maureen in the final scene , owing to a misunderstanding over a turtle that bites Maureen 's foot . Season 4 ( edit ) In season 4 it is
revealed that Suzanne 's crime was kidnapping and accidental manslaughter of a child she grew friendly with while working as a store greeter . Believing that
she and the boy were simply being friends , she did not realize the implications of her actions when she came across him playing in the park and invited him to
her apartment for ice cream and video games . The boy went willingly , but soon became frightened and wanted to call his dad . While Suzanne is out of the room
preparing for what she plans as more fun , he calls 911 . Suzanne becomes agitated as she tells him that 911 should only be called when there is an emergency .
The child panics and tries to get away . He goes onto the fire escape and falls to his death . It is mentioned that she took a plea deal .
</item 8>

<item 9>
Crazy Eyes (character)
Suzanne gives a statement to prison officials confessing to the attack , but inmate counselor Sam Healy ( Michael J. Harney ) produces evidence that exonerates
her . When she realizes that Vee used her , she breaks into tears . In episode 3 , `` Hugs Can Be Deceiving '' , Suzanne 's backstory is revealed via a
flashback . She was adopted by white parents and raised in the suburbs , but her mental instability alienated her classmates and their parents alike . Her
adoptive mother pushed her to excel at everything she did to prove that she was as good as everyone else ; ironically , the constant pressure only made Suzanne
's psychological problems worse . In the same episode , it is revealed that Suzanne had run outside and knocked Chapman unconscious after Chapman had beaten
Tiffany `` Pennsatucky '' Doggett ( Taryn Manning ) unconscious at the end of season 1 . Season 3 ( edit ) In season 3 , Taystee has her hands full trying to
control Suzanne , who is still grieving Vee 's death and is unstable , prone to losing her temper and constantly fighting with Poussey .
</item 9>

<item 10>
Graph query row:
entity: Suzanne Warren
relationship: RELATED_TO
detail: ex-girlfriend of
related: Maureen Kukudio
</item 10>
=== END ITEM id=34f24f38a90aa7fa ===

=== ITEM id=58f0430fa7f9cd95 ===
Gold sentences and facts:
Sentence 0: The original start date was January 2002 , but was pushed to February 7 in Los Angeles , California . Locations included Burbank , Downey , New York City , LA / Ontario International Airport ( which doubled for Miami International Airport ) , Quebec City and Montreal . The film was shot in 147 locations in only 52 days . DiCaprio reflected , `` Scenes that we thought would take three days took an afternoon '' . Filming ran from April 25 -- 30 in Park Avenue , just outside the Waldorf - Astoria Hotel . Production moved to Orange , New Jersey and returned to Brooklyn for bank and courthouse scenes . Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport . Quebec City was chosen for its European character and French feel . Place Royale , within Old Quebec , stands for Montrichard -- the church in the background of the arrest scene is Notre - Dame - des - Victoires . Filming ended on May 12 in Montreal .
  Fact 0.0: The original start date was January 2002.
  Fact 0.1: The start date was pushed to February 7 in Los Angeles, California.
  Fact 0.2: Locations included Burbank.
  Fact 0.3: Locations included Downey.
  Fact 0.4: Locations included New York City.
  Fact 0.5: Locations included LA / Ontario International Airport.
  Fact 0.6: LA / Ontario International Airport doubled for Miami International Airport.
  Fact 0.7: Locations included Quebec City.
  Fact 0.8: Locations included Montreal.
  Fact 0.9: The film was shot in 147 locations.
  Fact 0.10: The film was shot in only 52 days.
  Fact 0.11: DiCaprio reflected that scenes thought to take three days took an afternoon.
  Fact 0.12: Filming ran from April 25 to 30 in Park Avenue.
  Fact 0.13: Park Avenue filming was just outside the Waldorf-Astoria Hotel.
  Fact 0.14: Production moved to Orange, New Jersey.
  Fact 0.15: Production returned to Brooklyn for bank and courthouse scenes.
  Fact 0.16: Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport.
  Fact 0.17: Quebec City was chosen for its European character.
  Fact 0.18: Quebec City was chosen for its French feel.
  Fact 0.19: Place Royale, within Old Quebec, stands for Montrichard.
  Fact 0.20: The church in the background of the arrest scene is Notre-Dame-des-Victoires.
  Fact 0.21: Filming ended on May 12 in Montreal.
Sentence 1: Catch Me If You Can is a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson . The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen , and Nathalie Baye in supporting roles .
  Fact 1.0: Catch Me If You Can is a 2002 American biographical crime film.
  Fact 1.1: Catch Me If You Can was directed by Steven Spielberg.
  Fact 1.2: Catch Me If You Can was co-produced by Steven Spielberg.
  Fact 1.3: Catch Me If You Can has a screenplay by Jeff Nathanson.
  Fact 1.4: The film is based on the life of Frank Abagnale.
  Fact 1.5: Frank Abagnale successfully performed cons worth millions of dollars before his 19th birthday.
  Fact 1.6: Frank Abagnale performed cons by posing as a Pan American World Airways pilot.
  Fact 1.7: Frank Abagnale performed cons by posing as a Georgia doctor.
  Fact 1.8: Frank Abagnale performed cons by posing as a Louisiana parish prosecutor.
  Fact 1.9: Abagnale's primary crime was check fraud.
  Fact 1.10: Abagnale became so experienced that the FBI eventually turned to him for help.
  Fact 1.11: The FBI turned to Abagnale for help in catching other checking forgers.
  Fact 1.12: The film stars Leonardo DiCaprio.
  Fact 1.13: The film stars Tom Hanks.
  Fact 1.14: Christopher Walken, Martin Sheen, and Nathalie Baye have supporting roles in the film.
Sentence 2: Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received positively by both critics and Abagnale .
  Fact 2.0: Development for the film started in 1980.
  Fact 2.1: Development did not progress until 1997.
  Fact 2.2: In 1997, the film rights to Abagnale's book were sold to Spielberg's DreamWorks.
  Fact 2.3: David Fincher was a possible candidate for director before Spielberg decided to direct.
  Fact 2.4: Gore Verbinski was a possible candidate for director before Spielberg decided to direct.
  Fact 2.5: Lasse Hallström was a possible candidate for director before Spielberg decided to direct.
  Fact 2.6: Miloš Forman was a possible candidate for director before Spielberg decided to direct.
  Fact 2.7: Cameron Crowe was a possible candidate for director before Spielberg decided to direct.
  Fact 2.8: Filming took place from February to May 2002.
  Fact 2.9: The film was a financial success.
  Fact 2.10: The film was a critical success.
  Fact 2.11: The film was received positively by both critics and Abagnale.

Retrieved items:
<item 1>
Catch Me If You Can
Williams also earned a Grammy Award nomination . Elements of the film were later parodied in The Simpsons episode `` Catch ' Em If You Can '' . Home media (
edit ) Catch Me If You Can was released on DVD on May 6 , 2003 and on Blu - ray on December 4 , 2012 . Musical adaptation ( edit ) Main article : Catch Me If
You Can ( musical ) A musical adaptation of the same name premiered at the 5th Avenue Theatre in Seattle , Washington in July 2009 , starring Aaron Tveit and
Norbert Leo Butz . It began previews on Broadway at the Neil Simon Theatre on March 11 , 2011 and officially opened April 10 , 2011 . The musical was nominated
for four Tony Awards , including Best Musical . See also ( edit ) The Great Impostor , a 1961 film based on the story of an impostor named Ferdinand Waldo
Demara . The Pretender , a TV series References ( edit ) Jump up ^ `` CATCH ME IF YOU CAN ( 12A ) '' . British Board of Film Classification . December 13 , 2002
.
</item 1>

<item 2>
Catch Me If You Can
Jump up ^ Josef Adalian ( December 10 , 2002 ) . `` Inside Move : Net game for movie link '' . Variety . Retrieved June 30 , 2008 . Jump up ^ `` H'w 'd plays '
Catch ' '' . Variety . December 18 , 2002 . Retrieved June 30 , 2008 . Jump up ^ `` Catch Me If You Can ( 2002 ) '' . Box Office Mojo . Internet Movie Database
. Retrieved June 29 , 2008 . Jump up ^ `` 2002 Yearly Box Office Results '' . Box Office Mojo . Internet Movie Database . Retrieved June 29 , 2008 . Jump up ^
`` Catch Me If You Can '' . Rotten Tomatoes . Flixter . Retrieved June 29 , 2008 . Jump up ^ Ebert , Roger . `` Catch Me If You Can '' . Chicago Sun - Times .
Retrieved July 3 , 2008 . Jump up ^ Mick LaSalle ( December 25 , 2002 ) .
</item 2>

<item 3>
Catch Me If You Can
Jump up ^ Mick LaSalle ( December 25 , 2002 ) . `` Holiday Movies '' . San Francisco Chronicle . Retrieved July 3 , 2008 . Jump up ^ Stephen Hunter ( December
25 , 2002 ) . `` A Merry Chase '' . The Washington Post . Retrieved July 3 , 2008 . Jump up ^ Berardinelli , James . `` Catch Me If You Can '' . ReelViews.net .
Retrieved July 3 , 2008 . Jump up ^ Travers , Peter ( January 2 , 2003 ) . `` Catch Me If You Can '' . Rolling Stone . Archived from the original on April 19 ,
2008 . Retrieved July 3 , 2008 . Jump up ^ `` 74th Academy Awards '' . Internet Movie Database . Retrieved July 3 , 2008 . Jump up ^ `` 56th BAFTA Awards '' .
Internet Movie Database . Retrieved July 3 , 2008 . Jump up ^ `` Golden Globes : 2003 '' . Internet Movie Database .
</item 3>

<item 4>
Catch Me If You Can
Frank Abagnale , Jr. and Stan Redding . Catch Me If You Can : The Amazing True Story of the Youngest and Most Daring Con Man in the History of Fun and Profit .
( ISBN 0 - 06 - 052971 - 7 ) . External links ( edit ) Wikiquote has quotations related to : Catch Me If You Can Catch Me If You Can on IMDb Catch Me If You Can
at AllMovie Catch Me If You Can at the TCM Movie Database Catch Me If You Can at the American Film Institute Catalog Catch Me If You Can at Box Office Mojo
Catch Me If You Can at Rotten Tomatoes Catch Me If You Can at Metacritic An Interview with Leonardo DiCaprio - Movies Feature at IGN Official website for Catch
Me If You Can the musical ( hide ) Steven Spielberg Filmography Awards and nominations Films directed Firelight ( 1964 ) Slipstream ( 1967 ) Amblin ' ( 1968 )
Night Gallery ( `` Eyes '' segment , 1969 ) L.A.
</item 4>

<item 5>
Catch Me If You Can
Catch Me If You Can - wikipedia Catch Me If You Can Jump to : navigation , search This article is about the 2002 film . For other uses , see Catch Me If You Can
( disambiguation ) . Catch Me If You Can Theatrical release poster Directed by Steven Spielberg Produced by Steven Spielberg Walter F. Parkes Screenplay by Jeff
Nathanson Based on Catch Me If You Can by Frank Abagnale and Stan Redding Starring Leonardo DiCaprio Tom Hanks Music by John Williams Cinematography Janusz
Kamiński Edited by Michael Kahn Production company Amblin Entertainment Parkes / MacDonald Productions Distributed by DreamWorks Pictures Release date December
25 , 2002 ( 2002 - 12 - 25 ) Running time 141 minutes Country United States Language English Budget $52 million Box office $352.1 million Catch Me If You Can is
a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson .
</item 5>

<item 6>
Catch Me If You Can
The marketing department was careful to market the film as `` inspired by a true story '' in order to avoid controversy similar to that surrounding A Beautiful
Mind ( 2001 ) and The Hurricane ( 1999 ) , both of which deviated from history . The premiere took place at Westwood , Los Angeles , California on December 18 ,
2002 . Box Office ( edit ) Catch Me If You Can was released on December 25 , 2002 , earning slightly above $30 million in 3,225 theaters during its opening
weekend . The film went on to gross $164.6 million in North America and $187.5 million in foreign countries , coming at a worldwide total of $352.1 million .
The film was a financial success , recouping the $52 million budget six times over . Catch Me If You Can was the eleventh highest - grossing film of 2002 .
Minority Report ( also directed by Spielberg ) was tenth highest . Critical response ( edit ) On Rotten Tomatoes , the film has a certified fresh rating of 96 %
, based on 197 reviews , with an average rating of 7.9 / 10 .
</item 6>

<item 7>
Catch Me If You Can
wikipedia.org/w/index.php?title=Catch_Me_If_You_Can&oldid=821436887 '' Categories : 2002 films English - language films 2000s biographical films 2000s crime
drama films Amblin Entertainment films American biographical films American chase films American crime drama films American films Biographical films about
fraudsters Crime films based on actual events Drama films based on actual events DreamWorks Pictures films Films scored by John Williams Films about con artists
Films about identity theft Films based on biographies Films directed by Steven Spielberg Films set in Atlanta Films set in California Films set in Florida Films
set in France Films set in Louisiana Films set in Westchester County , New York Films set in New York City Films set in the 1960s Films shot in California Films
shot in Montreal Films produced by Steven Spielberg True crime films Fraud in fiction Hidden categories : Webarchive template wayback links Use mdy dates from
October 2015 Good articles Wikipedia articles with GND identifiers Talk Contents About Wikipedia Wikiquote বাংলা Български Català Čeština Dansk Deutsch Español
Euskara فارسی Français Galego 한국어
</item 7>

<item 8>
Catch Me If You Can
British Board of Film Classification . December 13 , 2002 . Retrieved February 13 , 2016 . Jump up ^ Van Luling , Todd ( October 17 , 2014 ) . `` 11 Easter Eggs
You Never Noticed In Your Favorite Movies '' . The Huffington Post . TheHuffingtonPost.com , Inc . Retrieved January 29 , 2015 . ^ Jump up to : Frank Abagnale (
September 3 , 2002 ) . `` Comments '' . Abagnale & Associates . Archived from the original on February 16 , 2009 . Retrieved June 30 , 2008 . ^ Jump up to :
Claude Brodesser ; Dana Harris ( August 21 , 2001 ) . `` D'Works to play ' Catch ' '' . Variety . Retrieved June 30 , 2008 . Jump up ^ Charles Lyons ; Dade
Hayes ( January 8 , 2002 ) . `` D'Works sets play date for pricey ' Catch ' '' . Variety . Retrieved June 29 , 2008 .
</item 8>

<item 9>
Catch Me If You Can
The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan
American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI
eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen ,
and Nathalie Baye in supporting roles . Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book
were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for
director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received
positively by both critics and Abagnale .
</item 9>

<item 10>
Catch Me If You Can
Locations included Burbank , Downey , New York City , LA / Ontario International Airport ( which doubled for Miami International Airport ) , Quebec City and
Montreal . The film was shot in 147 locations in only 52 days . DiCaprio reflected , `` Scenes that we thought would take three days took an afternoon '' .
Filming ran from April 25 -- 30 in Park Avenue , just outside the Waldorf - Astoria Hotel . Production moved to Orange , New Jersey and returned to Brooklyn for
bank and courthouse scenes . Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport . Quebec City was chosen for its
European character and French feel . Place Royale , within Old Quebec , stands for Montrichard -- the church in the background of the arrest scene is Notre -
Dame - des - Victoires . Filming ended on May 12 in Montreal . Music ( edit ) Main article : Catch Me If You Can ( soundtrack ) The film 's soundtrack was
released on December 10 , 2002 by DreamWorks Records . The original score was composed and conducted by John Williams .
</item 10>
=== END ITEM id=58f0430fa7f9cd95 ===

=== ITEM id=213dae986550f188 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins , Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone call , but her phone short circuits . Will 's friends Mike , Dustin , and Lucas search for Will in the woods and find Eleven . `` Chapter Two : The Weirdo on Maple Street '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) The boys bring Eleven to Mike 's house , where she sleeps in the basement . The next day , Mike nicknames her `` El '' . She says that `` bad people '' are looking for her and refuses to meet Mike 's parents . Scientists from the laboratory find a substance oozing from the walls of Joyce 's home . At Mike 's home , Eleven recognizes and points out Will in a photo . Dustin and Lucas want to inform Mike 's parents about Eleven , but she uses psychokinesis to stop them . While searching for Will , Mr. Clarke , the boys ' science teacher , discovers a scrap of a hospital gown outside the laboratory grounds . Nancy goes with her friend Barb to a party at her boyfriend Steve 's house . Will 's brother Jonathan investigates the woods where Will went missing . Hearing screaming , he runs to help but finds only Steve , Nancy , and their friends roughhousing around Steve 's swimming pool . He secretly photographs them . Barb , left alone by the poolside , vanishes . Joyce receives another call from Will , hears music from his room , and sees something coming through the wall . `` Chapter Three : Holly , Jolly '' Shawn Levy Jessica Mecklenburg July 15 , 2016 ( 2016 - 07 - 15 ) Barb wakes up in an empty and decaying pool . She screams for Nancy and attempts to climb out but is dragged down by something unseen . Joyce strings Christmas lights around her home to talk to Will , who can turn them on and off . Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab 's security camera footage . Hopper realizes it is fake . He and his deputies research Hawkins Lab , Brenner , and a woman named Terry Ives who claimed her daughter was taken by Brenner . Eleven has a flashback in which Brenner , whom she calls `` Papa , '' has her put in solitary confinement for refusing to telekinetically harm a cat . Nancy worries about Barb , who is missing . Steve and his friends discover Jonathan 's photographs . They destroy them and his camera . Nancy recovers a photo of Barb by the poolside and returns to Steve 's house to search for her . Nancy finds Barb 's car and sees the creature in the woods . Joyce establishes a code with Will using the lights , which he uses to tell her that he is alive but unsafe . He tells Joyce to run as a creature begins to climb through her wall . Will 's apparent body is discovered in the water at a quarry . `` Chapter Four : The Body '' Shawn Levy Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Eleven proves to the boys that Will is still alive by making contact through Mike 's walkie - talkie . Joyce goes to the morgue to view the body and refuses to believe it is Will . The boys give Eleven a makeover so they can sneak her into school . Joyce hears Will in her living room wall . Tearing away the wallpaper , she sees him behind a membrane . At school , Eleven uses Mr. Clarke 's ham radio to channel Will talking to his mother . Joyce demolishes the wall with an axe but reveals only her front porch . Nancy discovers a figure behind Barb in Jonathan 's photo . Jonathan realizes that the figure matches his mother 's description of the creature : a humanoid figure with long arms and no face . Nancy tells the police about her missing friend and subsequently admits to her mother that she slept with Steve . When she tells Steve about the police , she is irritated that he cares only about not getting in trouble with his father . Hopper confronts the state trooper who found Will 's body and beats him until he admits he was ordered to lie . Hopper goes to the morgue and finds that Will 's supposed body is a dummy . He breaks into Hawkins laboratory . 5 5 `` Chapter Five : The Flea and the Acrobat '' The Duffer Brothers Alison Tatlock July 15 , 2016 ( 2016 - 07 - 15 ) Will 's father Lonnie assures Joyce her experiences are hallucinations . Hopper discovers a portal in the basement of the lab but is knocked out by the lab 's guards . He wakes at home and discovers a hidden microphone . Sheriff 's deputies inform Hopper that Barb 's car was found , but Hopper finds the state 's involvement suspicious . The boys determine Will is trapped in an alternate dimension Eleven calls the `` Upside Down , '' and he is being pursued by a monster similar to a Demogorgon . After Will 's funeral , science teacher Mr. Clarke tells the boys that a high - energy spacetime tear could create a passage between dimensions . Using their compasses , they search for a disruption in the Earth 's electromagnetic field . Eleven remembers encountering the creature during a lab experiment , and afraid , she redirects the compasses . Lucas notices and confronts her . Mike defends her , he and Lucas fight , and Eleven telekinetically flings Lucas off Mike . While Dustin and Mike tend to the unconscious Lucas , Eleven disappears . In the woods , Nancy and Jonathan find a wounded deer . The creature drags the deer away , and Nancy follows the trail of blood to a portal to the Upside Down . She crawls through and discovers the creature feasting upon the deer . She snaps a twig and draws the creature 's attention . 6 6 `` Chapter Six : The Monster '' The Duffer Brothers Jessie Nickson - Lopez July 15 , 2016 ( 2016 - 07 - 15 ) Jonathan pulls Nancy back through the portal and saves her from the monster . Later , Steve sees them together through her bedroom window and assumes they are dating . Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store . During a fist fight with Steve , Jonathan inadvertently assaults a police officer and is arrested . Joyce and Hopper track down Terry Ives , who is catatonic and tended by her sister Becky . Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant ; she believed her daughter Jane was kidnapped by Brenner at birth . Joyce and Hopper conclude Eleven is likely Jane . Eleven shoplifts Eggo waffles . She recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement . While searching for Eleven , Mike and Dustin are ambushed by school bullies Troy and James . Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will 's body was discovered . Mike jumps but is levitated to safety by Eleven , who scares away the bullies . Lucas realizes agents from the laboratory are heading to Mike 's house . 7 7 `` Chapter Seven : The Bathtub '' The Duffer Brothers Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Lucas warns Mike via walkie - talkie that government agents are en route . Mike , Dustin , and Eleven flee . Eleven telekinetically flips a van that blocks their path , and the kids escape . Lucas reconciles with Mike and Eleven . Joyce and Hopper are called to the police station because of Jonathan 's assault . Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper . The group contacts Mike and his friends and they rendezvous . They ask Eleven to search for Will and Barb , but her earlier feats have left her too weak to find them . To amplify Eleven 's powers , they break into the middle school and build a sensory - deprivation tank . Eleven finds Barb 's corpse with a slug - like creature crawling out of her mouth . With Joyce 's help , she finds Will alive , hiding in the Upside Down `` Castle Byers , '' his backyard fort . Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards . Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station . In the Upside Down , the monster breaks into the fort where Will is hiding . 8 8 `` Chapter Eight : The Upside Down '' The Duffer Brothers Story by : Paul Dichter Teleplay by : The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) Hopper gives up Eleven 's location so that he and Joyce can enter the Upside Down . They find Will unconscious in the Upside - Down town library . They pull a tendril out of his throat , perform CPR , and revive him . Nancy and Jonathan booby - trap the Byers ' home , then cut their hands to attract the monster with their blood . Steve arrives , intending to apologize to Jonathan about their fight , but the monster attacks . The monster springs the trap but escapes back to the Upside Down . Eleven and the boys hide in the middle school . Mike asks Eleven to a school dance , the Snow Ball , and kisses her . The military storms the school to re-capture Eleven , but she kills them and collapses . As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends . Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from his family .
  Fact 0.0: The table lists, for each episode, its overall number, its number in the season, its title, its director, its writers, and its original release date.
  Fact 0.1: Chapter One: The Vanishing of Will Byers was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.2: Chapter One: The Vanishing of Will Byers was released on July 15, 2016.
  Fact 0.3: On November 6, 1983, in a U.S. Department of Energy laboratory in the town of Hawkins, Indiana, a scientist is attacked by an unseen creature.
  Fact 0.4: While bicycling home from a Dungeons & Dragons session with his friends, 12-year-old Will Byers encounters the creature and vanishes.
  Fact 0.5: The next day, a buzz-cut young girl wearing a hospital gown steals food from a local diner.
  Fact 0.6: The diner's owner, Benny, takes pity on the girl and feeds her before calling social services.
  Fact 0.7: From a tattoo on her arm, Benny learns that the girl's name is Eleven.
  Fact 0.8: A woman posing as a social worker arrives and murders Benny.
  Fact 0.9: Armed men search the diner for Eleven, but she escapes.
  Fact 0.10: Will's mother Joyce believes she hears Will's voice on a distorted phone call, but her phone short circuits.
  Fact 0.11: Will's friends Mike, Dustin, and Lucas search for Will in the woods and find Eleven.
  Fact 0.12: Chapter Two: The Weirdo on Maple Street was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.13: Chapter Two: The Weirdo on Maple Street was released on July 15, 2016.
  Fact 0.14: The boys bring Eleven to Mike's house, where she sleeps in the basement.
  Fact 0.15: The next day, Mike nicknames her "El".
  Fact 0.16: Eleven says that "bad people" are looking for her and refuses to meet Mike's parents.
  Fact 0.17: Scientists from the laboratory find a substance oozing from the walls of Joyce's home.
  Fact 0.18: At Mike's home, Eleven recognizes and points out Will in a photo.
  Fact 0.19: Dustin and Lucas want to inform Mike's parents about Eleven, but she uses psychokinesis to stop them.
  Fact 0.20: While searching for Will, Mr. Clarke, the boys' science teacher, discovers a scrap of a hospital gown outside the laboratory grounds.
  Fact 0.21: Nancy goes with her friend Barb to a party at her boyfriend Steve's house.
  Fact 0.22: Will's brother Jonathan investigates the woods where Will went missing.
  Fact 0.23: Hearing screaming, Jonathan runs to help but finds only Steve, Nancy, and their friends roughhousing around Steve's swimming pool.
  Fact 0.24: Jonathan secretly photographs Steve, Nancy, and their friends.
  Fact 0.25: Barb, left alone by the poolside, vanishes.
  Fact 0.26: Joyce receives another call from Will, hears music from his room, and sees something coming through the wall.
  Fact 0.27: Chapter Three: Holly, Jolly was directed by Shawn Levy and written by Jessica Mecklenburg.
  Fact 0.28: Chapter Three: Holly, Jolly was released on July 15, 2016.
  Fact 0.29: Barb wakes up in an empty and decaying pool.
  Fact 0.30: Barb screams for Nancy and attempts to climb out but is dragged down by something unseen.
  Fact 0.31: Joyce strings Christmas lights around her home to talk to Will, who can turn them on and off.
  Fact 0.32: Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab's security camera footage.
  Fact 0.33: Hopper realizes the footage is fake.
  Fact 0.34: Hopper and his deputies research Hawkins Lab, Brenner, and a woman named Terry Ives, who claimed her daughter was taken by Brenner.
  Fact 0.35: Eleven has a flashback in which Brenner, whom she calls "Papa," has her put in solitary confinement for refusing to telekinetically harm a cat.
  Fact 0.36: Nancy worries about Barb, who is missing.
  Fact 0.37: Steve and his friends discover Jonathan's photographs.
  Fact 0.38: Steve and his friends destroy Jonathan's photographs and his camera.
  Fact 0.39: Nancy recovers a photo of Barb by the poolside and returns to Steve's house to search for her.
  Fact 0.40: Nancy finds Barb's car and sees the creature in the woods.
  Fact 0.41: Joyce establishes a code with Will using the lights, which he uses to tell her that he is alive but unsafe.
  Fact 0.42: Will tells Joyce to run as a creature begins to climb through her wall.
  Fact 0.43: Will's apparent body is discovered in the water at a quarry.
  Fact 0.44: Chapter Four: The Body was directed by Shawn Levy and written by Justin Doble.
  Fact 0.45: Chapter Four: The Body was released on July 15, 2016.
  Fact 0.46: Eleven proves to the boys that Will is still alive by making contact through Mike's walkie-talkie.
  Fact 0.47: Joyce goes to the morgue to view the body and refuses to believe it is Will.
  Fact 0.48: The boys give Eleven a makeover so they can sneak her into school.
  Fact 0.49: Joyce hears Will in her living room wall.
  Fact 0.50: Tearing away the wallpaper, Joyce sees Will behind a membrane.
  Fact 0.51: At school, Eleven uses Mr. Clarke's ham radio to channel Will talking to his mother.
  Fact 0.52: Joyce demolishes the wall with an axe but reveals only her front porch.
  Fact 0.53: Nancy discovers a figure behind Barb in Jonathan's photo.
  Fact 0.54: Jonathan realizes the figure matches his mother's description of the creature: a humanoid figure with long arms and no face.
  Fact 0.55: Nancy tells the police about her missing friend.
  Fact 0.56: Nancy subsequently admits to her mother that she slept with Steve.
  Fact 0.57: When Nancy tells Steve about the police, she is irritated that he cares only about not getting in trouble with his father.
  Fact 0.58: Hopper confronts the state trooper who found Will's body and beats him until he admits he was ordered to lie.
  Fact 0.59: Hopper goes to the morgue and finds that Will's supposed body is a dummy.
  Fact 0.60: Hopper breaks into Hawkins laboratory.
  Fact 0.61: Chapter Five: The Flea and the Acrobat was directed by The Duffer Brothers and written by Alison Tatlock.
  Fact 0.62: Chapter Five: The Flea and the Acrobat was released on July 15, 2016.
  Fact 0.63: Will's father Lonnie assures Joyce her experiences are hallucinations.
  Fact 0.64: Hopper discovers a portal in the basement of the lab but is knocked out by the lab's guards.
  Fact 0.65: Hopper wakes at home and discovers a hidden microphone.
  Fact 0.66: Sheriff's deputies inform Hopper that Barb's car was found.
  Fact 0.67: Hopper finds the state's involvement suspicious.
  Fact 0.68: The boys determine Will is trapped in an alternate dimension that Eleven calls the "Upside Down."
  Fact 0.69: The boys determine Will is being pursued by a monster similar to a Demogorgon.
  Fact 0.70: After Will's funeral, science teacher Mr. Clarke tells the boys that a high-energy spacetime tear could create a passage between dimensions.
  Fact 0.71: Using their compasses, the boys search for a disruption in the Earth's electromagnetic field.
  Fact 0.72: Eleven remembers encountering the creature during a lab experiment.
  Fact 0.73: Afraid, Eleven redirects the compasses.
  Fact 0.74: Lucas notices and confronts Eleven.
  Fact 0.75: Mike defends Eleven, and he and Lucas fight.
  Fact 0.76: Eleven telekinetically flings Lucas off Mike.
  Fact 0.77: While Dustin and Mike tend to the unconscious Lucas, Eleven disappears.
  Fact 0.78: In the woods, Nancy and Jonathan find a wounded deer.
  Fact 0.79: The creature drags the deer away, and Nancy follows the trail of blood to a portal to the Upside Down.
  Fact 0.80: Nancy crawls through the portal and discovers the creature feasting upon the deer.
  Fact 0.81: Nancy snaps a twig and draws the creature's attention.
  Fact 0.82: Chapter Six: The Monster was directed by The Duffer Brothers and written by Jessie Nickson-Lopez.
  Fact 0.83: Chapter Six: The Monster was released on July 15, 2016.
  Fact 0.84: Jonathan pulls Nancy back through the portal and saves her from the monster.
  Fact 0.85: Later, Steve sees Nancy and Jonathan together through her bedroom window and assumes they are dating.
  Fact 0.86: Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store.
  Fact 0.87: During a fist fight with Steve, Jonathan inadvertently assaults a police officer and is arrested.
  Fact 0.88: Joyce and Hopper track down Terry Ives, who is catatonic and tended by her sister Becky.
  Fact 0.89: Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant.
  Fact 0.90: Becky explains that Terry believed her daughter Jane was kidnapped by Brenner at birth.
  Fact 0.91: Joyce and Hopper conclude Eleven is likely Jane.
  Fact 0.92: Eleven shoplifts Eggo waffles.
  Fact 0.93: Eleven recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement.
  Fact 0.94: While searching for Eleven, Mike and Dustin are ambushed by school bullies Troy and James.
  Fact 0.95: Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will's body was discovered.
  Fact 0.96: Mike jumps but is levitated to safety by Eleven, who scares away the bullies.
  Fact 0.97: Lucas realizes agents from the laboratory are heading to Mike's house.
  Fact 0.98: Chapter Seven: The Bathtub was directed by The Duffer Brothers and written by Justin Doble.
  Fact 0.99: Chapter Seven: The Bathtub was released on July 15, 2016.
  Fact 0.100: Lucas warns Mike via walkie-talkie that government agents are en route.
  Fact 0.101: Mike, Dustin, and Eleven flee.
  Fact 0.102: Eleven telekinetically flips a van that blocks their path, and the kids escape.
  Fact 0.103: Lucas reconciles with Mike and Eleven.
  Fact 0.104: Joyce and Hopper are called to the police station because of Jonathan's assault.
  Fact 0.105: Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper.
  Fact 0.106: The group contacts Mike and his friends, and they rendezvous.
  Fact 0.107: The group asks Eleven to search for Will and Barb, but her earlier feats have left her too weak to find them.
  Fact 0.108: To amplify Eleven's powers, the group breaks into the middle school and builds a sensory-deprivation tank.
  Fact 0.109: Eleven finds Barb's corpse with a slug-like creature crawling out of her mouth.
  Fact 0.110: With Joyce's help, Eleven finds Will alive, hiding in the Upside Down "Castle Byers," his backyard fort.
  Fact 0.111: Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards.
  Fact 0.112: Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station.
  Fact 0.113: In the Upside Down, the monster breaks into the fort where Will is hiding.
  Fact 0.114: Chapter Eight: The Upside Down was directed by The Duffer Brothers, with story by Paul Dichter and teleplay by The Duffer Brothers.
  Fact 0.115: Chapter Eight: The Upside Down was released on July 15, 2016.
  Fact 0.116: Hopper gives up Eleven's location so that he and Joyce can enter the Upside Down.
  Fact 0.117: Hopper and Joyce find Will unconscious in the Upside-Down town library.
  Fact 0.118: Hopper and Joyce pull a tendril out of Will's throat, perform CPR, and revive him.
  Fact 0.119: Nancy and Jonathan booby-trap the Byers' home, then cut their hands to attract the monster with their blood.
  Fact 0.120: Steve arrives, intending to apologize to Jonathan about their fight, but the monster attacks.
  Fact 0.121: The monster springs the trap but escapes back to the Upside Down.
  Fact 0.122: Eleven and the boys hide in the middle school.
  Fact 0.123: Mike asks Eleven to a school dance, the Snow Ball, and kisses her.
  Fact 0.124: The military storms the school to re-capture Eleven, but she kills them and collapses.
  Fact 0.125: As Brenner cradles Eleven, the monster attacks, apparently killing him.
  Fact 0.126: The boys hide Eleven in a classroom.
  Fact 0.127: The monster corners the boys and Eleven, but Eleven pins it against a wall.
  Fact 0.128: Eleven says goodbye to Mike, disintegrates the monster, and vanishes.
  Fact 0.129: Will is hospitalized and reunited with his mother, brother, and friends.
  Fact 0.130: Hopper is reluctantly picked up by a black car.
  Fact 0.131: One month later, Nancy and Steve are back together, and both are friends with Jonathan.
  Fact 0.132: Hopper leaves Eggo waffles and other food in a box in the woods.
  Fact 0.133: Will coughs up a slug-like creature and has a vision of the world as the Upside Down, but he hides this from his family.

Retrieved items:
<item 1>
Stranger Things
Eleven visualizes Mike and Hopper and realizes they are in trouble . While Kali and her gang escape the police , Eleven returns to Hawkins . 16 8 `` Chapter
Eight : The Mind Flayer '' The Duffer Brothers The Duffer Brothers October 27 , 2017 ( 2017 - 10 - 27 ) The Demodogs overwhelm the lab . Mike convinces Joyce to
sedate Will so that the Shadow Monster can not track them . Mike , Will , Joyce , Hopper , Bob , and Owens shelter in the lab 's security room . Dustin , Steve
, Lucas , and Max track the Demodogs to the lab , where they regroup with Nancy and Jonathan . When the power goes out , Bob resets the circuit breakers so they
can escape . Owens stays behind and uses the security cameras to guide the others out . Mike , Will , Joyce , and Hopper escape , but Bob is killed by the
Demodogs . The group retreat to the Byers ' house , where Dustin suggests that the Shadow Monster is a Mind Flayer . After disguising their location , they
interrogate Will .
</item 1>

<item 2>
Stranger Things
Episodes Season 1 ( 2016 ) No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The
Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins ,
Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers
encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes
pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker
arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone
call , but her phone short circuits .
</item 2>

<item 3>
Graph query row:
character: Benny
death_date: None
connected_to: Chapter One: The Vanishing of Will Byers
path_entities: ['Benny', 'local diner', None, 'Chapter One: The Vanishing of Will Byers']
relationship_types: ['RELATED_TO', 'FROM_CHUNK', 'FROM_CHUNK']
</item 3>

<item 4>
Stranger Things
Cast and characters The Duffer Brothers , Shawn Levy and the main cast of season 2 of Stranger Things at the 2017 San Diego Comic - Con International Main
Winona Ryder as Joyce Byers , the mother of Will and Jonathan Byers . She is divorced from Lonnie Byers . David Harbour as Jim Hopper , chief of Hawkins Police
Department . After his young daughter died of cancer , Hopper got divorced and lapsed into alcoholism . Finn Wolfhard as Mike Wheeler , son of Karen and Ted ,
younger brother of Nancy , and one of three friends of Will Byers . He is an intelligent and conscientious student and is committed to his friends . He develops
romantic feelings for Eleven . Millie Bobby Brown as Eleven ( `` El '' ) , a young girl with psychokinetic abilities and a limited vocabulary . Her real name is
Jane , the biological daughter of Terry Ives . After escaping from Hawkins Laboratory , where experiments were being performed on her , she befriends Mike ,
Dustin , and Lucas . She develops romantic feelings for Mike . Gaten Matarazzo as Dustin Henderson , one of Will 's friends .
</item 4>

<item 5>
Stranger Things
For the second season , Schnapp and Keery were promoted to series regulars , along with the additions of Sadie Sink , Dacre Montgomery , Sean Astin and Paul
Reiser . Set in the fictional town of Hawkins , Indiana , in the 1980s , the first season focuses on the investigation into the disappearance of a young boy
amid supernatural events occurring around the town including the appearance of a girl with psychokinetic abilities who helps the missing boy 's friends in their
own search . The second season , titled Stranger Things 2 , is set a year after the first , and deals with attempts of the characters to return to normal and
consequences that linger from the first season . The Duffer Brothers developed the series as a mix of investigative drama alongside supernatural elements with
childlike sensibilities , establishing its time frame in the 1980s and creating a homage to pop culture of that decade . Several themes and directorial aspects
were inspired and aesthetically informed by the works of Steven Spielberg , John Carpenter , and Stephen King , among others , including several films , anime
and video games . The first season was released on Netflix on July 15 , 2016 .
</item 5>

<item 6>
Stranger Things
Nancy and Jonathan are caught by lab agents . Owens shows them the gate to the Upside Down . He says it can be contained but not destroyed , admits Barb died ,
and allows Nancy and Jonathan to go free . Secretly , Nancy records his confession . Billy warns Max to stay away from Lucas . Eleven finds Hopper 's notes on
Hawkins Lab . Using her psychic powers , she contacts Terry Ives , who calls her `` Jane . '' Dustin discovers Dart devouring his cat and realizes Dart is a
young Demogorgon . Following Will 's map , Hopper digs in Merril 's field and finds a tunnel corrupted by the Upside Down . 13 5 `` Chapter Five : Dig Dug ''
Andrew Stanton Jessie Nickson - Lopez October 27 , 2017 ( 2017 - 10 - 27 ) Vines trap Hopper in the infected tunnels . Will realizes Hopper is in danger . Lucas
tells Max the truth about Will 's disappearance last year , but she does n't believe him . Jonathan and Nancy play Owens ' confession to Bauman .
</item 6>

<item 7>
Stranger Things
The boys give Eleven a makeover so they can sneak her into school . Joyce hears Will in her living room wall . Tearing away the wallpaper , she sees him behind
a membrane . At school , Eleven uses Mr. Clarke 's ham radio to channel Will talking to his mother . Joyce demolishes the wall with an axe but reveals only her
front porch . Nancy discovers a figure behind Barb in Jonathan 's photo . Jonathan realizes that the figure matches his mother 's description of the creature :
a humanoid figure with long arms and no face . Nancy tells the police about her missing friend and subsequently admits to her mother that she slept with Steve .
When she tells Steve about the police , she is irritated that he cares only about not getting in trouble with his father . Hopper confronts the state trooper
who found Will 's body and beats him until he admits he was ordered to lie . Hopper goes to the morgue and finds that Will 's supposed body is a dummy . He
breaks into Hawkins laboratory .
</item 7>

<item 8>
Stranger Things
In the Upside Down , the monster breaks into the fort where Will is hiding . 8 8 `` Chapter Eight : The Upside Down '' The Duffer Brothers Story by : Paul
Dichter Teleplay by : The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) Hopper gives up Eleven 's location so that he and Joyce can enter the Upside Down .
They find Will unconscious in the Upside - Down town library . They pull a tendril out of his throat , perform CPR , and revive him . Nancy and Jonathan booby -
trap the Byers ' home , then cut their hands to attract the monster with their blood . Steve arrives , intending to apologize to Jonathan about their fight ,
but the monster attacks . The monster springs the trap but escapes back to the Upside Down . Eleven and the boys hide in the middle school . Mike asks Eleven to
a school dance , the Snow Ball , and kisses her . The military storms the school to re-capture Eleven , but she kills them and collapses . As Brenner cradles
her , the monster attacks , apparently killing him .
</item 8>

<item 9>
Stranger Things
As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it
against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends
. Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo
waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from
his family . Season 2 ( 2017 ) No . overall No. in season Title Directed by Written by Original release date 9 `` Chapter One : MADMAX '' The Duffer Brothers
The Duffer Brothers October 27 , 2017 ( 2017 - 10 - 27 ) On October 28 , 1984 , Kali and her gang use her psychic powers to escape the police . Her wrist tattoo
marks her as `` 008 . ''
</item 9>

<item 10>
Stranger Things
Jonathan and Nancy play Owens ' confession to Bauman . The three agree to publicize a watered down and more believable story about Barb dying from toxins
released by the lab . Dustin traps Dart in his root cellar and recruits Steve to fight it . Owens and the lab scientists realize the soil from the infected
pumpkin patches is part of a hive mind . Eleven hitchhikes to Terry 's house . She telepathically communicates with Terry , seeing images of another child kept
in Hawkins Lab and of Terry being crippled by forced shock therapy . Bob figures out Hopper 's location , and he and Joyce enter the tunnels to rescue him .
They are joined by scientists from Hawkins Lab . As the scientists burn the vines filling the tunnels , Will screams in agony and collapses . 14 6 `` Chapter
Six : The Spy '' Andrew Stanton Kate Trefry October 27 , 2017 ( 2017 - 10 - 27 ) Will is rushed to the lab , where he can not recognize Bob , Hopper , or Owens
. Owens speculates that damaging the tunnels may be lethal to Will .
</item 10>
=== END ITEM id=213dae986550f188 ===

=== ITEM id=c6ad10f9a1d2ee44 ===
Gold sentences and facts:
Sentence 0: 2016 production of still wine State Production ( gal ) Production ( % ) Alabama 34,966 0.004 % Arizona 190,008 0.024 % Arkansas 246,363 0.031 % California 680,272,512 84.354 % Colorado 556,994 0.069 % Connecticut 134,517 0.017 % Florida 1,634,103 0.203 % Georgia 276,144 0.034 % Idaho 497,007 0.062 % Illinois 391,805 0.049 % Indiana 1,411,540 0.175 % Iowa 335,522 0.042 % Kansas 104,129 0.013 % Kentucky 2,176,059 0.270 % Louisiana 38,232 0.005 % Maine 48,222 0.006 % Maryland 436,185 0.054 % Massachusetts 792,884 0.098 % Michigan 2,576,238 0.319 % Minnesota 331,946 0.041 % Missouri 993,831 0.123 % Montana 33,445 0.004 % Nebraska 120,366 0.015 % New Hampshire 159,316 0.020 % New Jersey 1,832,325 0.227 % New Mexico 749,818 0.093 % New York 27,969,308 3.468 % North Carolina 1,903,060 0.236 % Ohio 5,938,738 0.736 % Oklahoma 70,204 0.009 % Oregon 11,822,972 1.466 % Pennsylvania 12,405,181 1.538 % South Carolina 77,842 0.010 % South Dakota 139,738 0.017 % Tennessee 1,279,752 0.159 % Texas 1,907,299 0.237 % Vermont 2,172,526 0.269 % Virginia 2,157,395 0.268 % Washington 40,747,190 5.053 % West Virginia 40,733 0.005 % Wisconsin 1,129,405 0.140 % Others 312,051 0.039 % Sum 806,447,891 100 %
  Fact 0.0: The table reports the 2016 production of still wine.
  Fact 0.1: Alabama produced 34,966 gallons of still wine in 2016, which is 0.004 percent of production.
  Fact 0.2: Arizona produced 190,008 gallons of still wine in 2016, which is 0.024 percent of production.
  Fact 0.3: Arkansas produced 246,363 gallons of still wine in 2016, which is 0.031 percent of production.
  Fact 0.4: California produced 680,272,512 gallons of still wine in 2016, which is 84.354 percent of production.
  Fact 0.5: Colorado produced 556,994 gallons of still wine in 2016, which is 0.069 percent of production.
  Fact 0.6: Connecticut produced 134,517 gallons of still wine in 2016, which is 0.017 percent of production.
  Fact 0.7: Florida produced 1,634,103 gallons of still wine in 2016, which is 0.203 percent of production.
  Fact 0.8: Georgia produced 276,144 gallons of still wine in 2016, which is 0.034 percent of production.
  Fact 0.9: Idaho produced 497,007 gallons of still wine in 2016, which is 0.062 percent of production.
  Fact 0.10: Illinois produced 391,805 gallons of still wine in 2016, which is 0.049 percent of production.
  Fact 0.11: Indiana produced 1,411,540 gallons of still wine in 2016, which is 0.175 percent of production.
  Fact 0.12: Iowa produced 335,522 gallons of still wine in 2016, which is 0.042 percent of production.
  Fact 0.13: Kansas produced 104,129 gallons of still wine in 2016, which is 0.013 percent of production.
  Fact 0.14: Kentucky produced 2,176,059 gallons of still wine in 2016, which is 0.270 percent of production.
  Fact 0.15: Louisiana produced 38,232 gallons of still wine in 2016, which is 0.005 percent of production.
  Fact 0.16: Maine produced 48,222 gallons of still wine in 2016, which is 0.006 percent of production.
  Fact 0.17: Maryland produced 436,185 gallons of still wine in 2016, which is 0.054 percent of production.
  Fact 0.18: Massachusetts produced 792,884 gallons of still wine in 2016, which is 0.098 percent of production.
  Fact 0.19: Michigan produced 2,576,238 gallons of still wine in 2016, which is 0.319 percent of production.
  Fact 0.20: Minnesota produced 331,946 gallons of still wine in 2016, which is 0.041 percent of production.
  Fact 0.21: Missouri produced 993,831 gallons of still wine in 2016, which is 0.123 percent of production.
  Fact 0.22: Montana produced 33,445 gallons of still wine in 2016, which is 0.004 percent of production.
  Fact 0.23: Nebraska produced 120,366 gallons of still wine in 2016, which is 0.015 percent of production.
  Fact 0.24: New Hampshire produced 159,316 gallons of still wine in 2016, which is 0.020 percent of production.
  Fact 0.25: New Jersey produced 1,832,325 gallons of still wine in 2016, which is 0.227 percent of production.
  Fact 0.26: New Mexico produced 749,818 gallons of still wine in 2016, which is 0.093 percent of production.
  Fact 0.27: New York produced 27,969,308 gallons of still wine in 2016, which is 3.468 percent of production.
  Fact 0.28: North Carolina produced 1,903,060 gallons of still wine in 2016, which is 0.236 percent of production.
  Fact 0.29: Ohio produced 5,938,738 gallons of still wine in 2016, which is 0.736 percent of production.
  Fact 0.30: Oklahoma produced 70,204 gallons of still wine in 2016, which is 0.009 percent of production.
  Fact 0.31: Oregon produced 11,822,972 gallons of still wine in 2016, which is 1.466 percent of production.
  Fact 0.32: Pennsylvania produced 12,405,181 gallons of still wine in 2016, which is 1.538 percent of production.
  Fact 0.33: South Carolina produced 77,842 gallons of still wine in 2016, which is 0.010 percent of production.
  Fact 0.34: South Dakota produced 139,738 gallons of still wine in 2016, which is 0.017 percent of production.
  Fact 0.35: Tennessee produced 1,279,752 gallons of still wine in 2016, which is 0.159 percent of production.
  Fact 0.36: Texas produced 1,907,299 gallons of still wine in 2016, which is 0.237 percent of production.
  Fact 0.37: Vermont produced 2,172,526 gallons of still wine in 2016, which is 0.269 percent of production.
  Fact 0.38: Virginia produced 2,157,395 gallons of still wine in 2016, which is 0.268 percent of production.
  Fact 0.39: Washington produced 40,747,190 gallons of still wine in 2016, which is 5.053 percent of production.
  Fact 0.40: West Virginia produced 40,733 gallons of still wine in 2016, which is 0.005 percent of production.
  Fact 0.41: Wisconsin produced 1,129,405 gallons of still wine in 2016, which is 0.140 percent of production.
  Fact 0.42: Other states produced 312,051 gallons of still wine in 2016, which is 0.039 percent of production.
  Fact 0.43: The total 2016 production of still wine was 806,447,891 gallons, which is 100 percent.

Retrieved items:
<item 1>
U.S. history of alcohol minimum purchase age by state
1934 : 21 21 21 21 New York 21 18 18 1982 : Raised to 19 in 1982 ( Dec ) 1985 : Raised to 21 in 1985 ( Dec ) 21 North Carolina ? 18 18 : for beer and wine 21 :
for liquor 1983 : 19 : for beer and wine 21 : for liquor 1986 : Raised to 21 in 1986 ( Sep 1 ) 21 North Dakota ? 1936 : 21 21 21 21 Ohio ?
</item 1>

<item 2>
U.S. history of alcohol minimum purchase age by state
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
U.S. history of alcohol minimum purchase age by state
That is because the federal law is concerned only with purchase and public possession , not private consumption , and contains several exceptions . As of
January 1 , 2007 , 14 states and the District of Columbia ban underage consumption outright , 19 states do not specifically ban underage consumption outright ,
and 17 states have family member or location exceptions to their underage consumption laws . Federal law explicitly provides for religious , medical ,
employment and private club possession exceptions ; as of 2005 , 31 states have family member or location exceptions to their underage possession laws . In the
1960s the age for buying or drinking beer and wine in the District of Columbia ( Washington , D.C. ) was 18 ; the age for hard liquor was 21 . Residents from
Virginia and Maryland would often drive to D.C. to obtain alcohol . In Louisiana , the 1987 law raising the age from 18 to 21 was deliberately written solely to
comply with the National Minimum Drinking Age Act of 1984 to avoid losing highway funding , while still allowing 18 - to 20 - year - olds to drink as before .
</item 3>

<item 4>
U.S. history of alcohol minimum purchase age by state
New York Times . Retrieved 2009 - 09 - 06 . ^ Jump up to : ( 2 ) Exceptions to Minimum Age of 21 for Possession of Alcohol as of January 1 , 2010 , Alcohol
Policy Information System Jump up ^ http://www.legis.la.gov/legis/Law.aspx?d=78725 ^ Jump up to : `` Drinking ages set by the states '' . New York Times . June
27 , 1984 . Jump up ^ The Years of the Rat . Magazine.loyola.edu ( 2009 - 08 - 18 ) . Retrieved on 2011 - 06 - 22 . Jump up ^ Effects of Drunk Driving .
Drivingdrunk.org . Retrieved on 2011 - 06 - 22 . Jump up ^ `` Maryland increases drinking age to 21 '' . Gettysburg Times . Associated Press . February 26 ,
1982 . p. 3 . Jump up ^ `` Legal drinking age going up '' . Eugene Register - Guard . Associated Press .
</item 4>

<item 5>
U.S. history of alcohol minimum purchase age by state
`` Compiled Statutes of New Jersey '' . Soney & Sage -- via Google Books . Jump up ^ `` THE REGION - Drinking Age of 21 Gains in Jersey '' . NYTimes.com . New
Jersey . 1982 - 06 - 11 . Retrieved 2016 - 09 - 10 . Jump up ^ Barbanel , Josh ( May 28 , 1984 ) . `` Raising New York Drinking Age to 21 '' . New York Times .
Jump up ^ `` Drinking Age of 21 Gains in Jersey '' . New York Times . June 11 , 1982 . Retrieved 2010 - 01 - 30 . `` New York will raise its legal age of 18 to
19 on Dec. 4 . Jump up ^ `` New York raising its drinking age to 21 in December '' . New York Times . June 19 , 1985 . Retrieved 2012 - 08 - 20 . The minimum
legal drinking age in New York State is rising from 19 to 21 on Dec. 1 .
</item 5>

<item 6>
Minimum wage in the United States
advocacy group ) United States state - related lists List of states and territories of the United States Demographics Educational attainment Irreligion Largest
cities by population Most popular given names Most populous counties Population Density Growth rate Birth and death rates Historical African American Amish
Asian Hispanic and Latino Spanish - speaking Religiosity Economy Billionaires Budgets Companies Federal tax revenue Federal taxation and spending Gross domestic
product Growth rate Per capita Income Household Per capita Inequality Minimum wages Poverty rates Sales taxes Sovereign wealth funds State income taxes Flat
rate None Unemployment rates Union affiliation Vehicles per capita Environment Carbon dioxide emissions Botanical gardens Parks Renewable energy Superfund sites
Wilderness areas Geography Area Coastline Elevation Extreme points Forest Geographic centers Regions Government Attorneys general Capitals Historical Capitol
buildings Comparison Counties Alphabetical Current State Legislators Governors Lieutenant governors Legislatures Libraries and archives Official languages Poets
laureate Political divisions State auditors State secretaries of state State superintendents of education State supreme courts State treasurers Health Human
Development Index Fertility rates Hospitals Life expectancy Obesity rates History Date of statehood Name etymologies Historical societies and museums National
Historic Landmarks National Register of Historic Places State partitions Historic regions Law Abortion Age of consent Alcohol Dry communities Alford plea Cell
phone use while driving Constitutions Firearms
</item 6>

<item 7>
U.S. history of alcohol minimum purchase age by state
1936 : 21 21 21 21 Ohio ? 1935 : 18 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 18 : for 3.2 % ABW beer 21 : for beer stronger
than 3.2 % ABW , wine and liquor 1982 : 19 : for 3.2 % ABW beer 21 : for beer stronger than 3.2 % ABW , wine and liquor 1982 : 19 : beer 21 : wine and liquor
1988 : 21 21 ( However , anyone can drink with parents or legal - age spouse ) Oklahoma ? Men : 21 Women : 18 : 3.2 % ABW beer 21 : for beer stronger than 3.2 %
ABW , wine and liquor 1976 : Lowered to 18 ( for 3.2 % ABW beer ) for both sexes in 1976 ( Dec ) Last state to lower the drinking age . 1983 : Raised to 21 21
Oregon ?
</item 7>

<item 8>
U.S. history of alcohol minimum purchase age by state
1 . ISSN 0148 - 7736 . Start : p. 74 . CITED : p. 76 . Jump up ^ `` Drinking at Midd : past and present '' . Media.www.middleburycampus.com . Retrieved 2009 -
09 - 06 . ^ Jump up to : `` Health Care in Vermont Database and Timeline '' ( PDF ) . Retrieved 2009 - 09 - 06 . ^ Jump up to : `` History of the Legal Drinking
Age '' . Abc.state.va.us. 1966 - 07 - 01 . Retrieved 2009 - 09 - 06 . Jump up ^ What was the first legal drinking age in Wash. ? . Blog.seattlepi.com ( 2009 -
11 - 09 ) . Retrieved on 2011 - 06 - 22 .
</item 8>

<item 9>
U.S. history of alcohol minimum purchase age by state
21 1974 : Lowered to 18 : for beer and wine ( 21 : for liquor remained ) 1981 : Raised to 19 for off - premises consumption 1983 : Raised to 19 ( all beer )
1985 : Raised to 21 21 Washington `` Age of majority '' since 1877 21 21 21 21 West Virginia ? 18 : for beer and wine 21 : for liquor 1972 : Lowered to 18 ( all
) 1983 : Raised to 19 ( & 21 for non-residents ) 1986 : Raised to 21 21 Wisconsin 1839 -- 1866 : 18 : for wine and liquor None : for beer Post-1866 : 21 ( all )
18 : for beer 21 : for wine and spirits 1957 : 21 : for residents of bordering states with age limit 21 . 1963 : Raised to 21 ( off - premises beer sales ,
remained 18 for on - premises ) .
</item 9>

<item 10>
U.S. history of alcohol minimum purchase age by state
1983 : Raised to 21 21 Oregon ? 1933 : 21 21 21 21 Pennsylvania ? 1935 : 21 21 21 21 Rhode Island ? 21 1970 : Lowered to 18 1980 : Raised to 19 1981 : Raised to
20 1984 : Raised to 21 21 South Carolina 21 1935 : 18 : for beer and wine 21 : for liquor 18 : for beer and wine 21 : for liquor 1984 : 19 : for beer and wine
21 : for liquor 1986 : Raised to 21 21 South Dakota ?
</item 10>
=== END ITEM id=c6ad10f9a1d2ee44 ===

=== ITEM id=a08ec3932a745a25 ===
Gold sentences and facts:
Sentence 0: Vaugier appeared in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
  Fact 0.0: Vaugier appeared in Need for Speed: Carbon as Nikki.
  Fact 0.1: Nikki is the girlfriend of the antagonist in Need for Speed: Carbon.
  Fact 0.2: Nikki is the love interest of the protagonist in Need for Speed: Carbon.
Sentence 1: Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice
  Fact 1.0: The filmography table lists a 2006 role as Nikki in Need for Speed: Carbon, with notes Voice, Mo-Cap, & Likeness.
  Fact 1.1: The filmography table lists a 2015 role as Jade in Dying Light, with notes Mo-Cap.
  Fact 1.2: The filmography table lists a 2015 role as Maine, The Female Kin in Hellraid, with notes Voice.

Retrieved items:
<item 1>
Emmanuelle Vaugier
She has an older brother , Jason Vaugier . Career ( edit ) Television ( edit ) Vaugier played the role of young , upstart model Maria Alcobar in a 1996 episode
of Highlander : The Series . In 2004 , she played Lindsey Kellogg on the short - lived Fox Television series North Shore . She has also had guest - star spots
on Veronica Mars , Supernatural , The Outer Limits , Higher Ground , Andromeda , and Charmed . She also appeared in the Emmy Award nominated miniseries The
Beach Boys : An American Family where she portrayed Mike Love 's wife , Suzanne . Vaugier also completed a starring role in MOW Veiled Truth for the Lifetime
Network . She is also known for her role during the first two seasons of One Tree Hill , as Nicki , Jake 's ( Bryan Greenberg ) ex-girlfriend and mother to
their child , Jenny . From 2005 - 2015 , Vaugier guest starred in Twelve episodes of CBS 's hit sitcom Two and a Half Men as Mia , a ballet teacher with whom
Charlie Harper fell in love and almost married .
</item 1>

<item 2>
Deadpool 2
`` ' Deadpool 2 ' Stunt Crash Victim ID 'd As First African - American Female Pro Road Racer ; Director & Fox `` Deeply Saddened '' -- Update `` . Deadline.com
. Archived from the original on August 15 , 2017 . Retrieved August 15 , 2017 . ^ Jump up to : Patten , Dominic ; Busch , Anita ( August 15 , 2017 ) . `` '
Deadpool 2 ' To Start Production 48 Hours After Stunt Driver Killed '' . Deadline.com . Archived from the original on August 16 , 2017 . Retrieved August 16 ,
2017 . Jump up ^ Johnson , Scott ( August 16 , 2017 ) . `` ' Deadpool 2 ' Motorcycle Death Leaves Stunt Community Asking Tough Questions '' . Deadline.com .
Archived from the original on August 17 , 2017 . Retrieved August 17 , 2017 . Jump up ^ Reynolds , Ryan ( @ VancityReynolds ) ( August 1 , 2017 ) .
</item 2>

<item 3>
Krysten Ritter
The The Last International Playboy Ozzy 2008 27 Dresses Gina the goth 2008 What Happens in Vegas Kelly 2009 Glock Beretta Short film 2009 Confessions of a
Shopaholic Suze Cleath - Stuart She 's Out of My League Patty How to Make Love to a Woman Lauren 2011 Killing Bono Gloria 2011 Life Happens Kim Also writer and
co-producer 2011 Margaret Shopgirl 2012 BuzzKill Nicole 2012 Vamps Stacy Daimen 2012 Refuge Amy 2014 Listen Up Philip Melanie 2014 Veronica Mars Gia Goodman
2014 Asthma Ruby 2014 Search Party Christy 2014 Big Eyes DeAnn 2017 Hero , The The Hero Lucy Television ( edit ) Year Title Role Notes Whoopi Brynn Episode : ``
The Squatters '' One Life to Live Kay 4 episodes Law & Order Tracy Warren Episode : `` Everybody Loves Raimondo 's '' Tanner on Tanner Saleswoman 2 episodes
2005 Jonny Zero Quinn Episode : `` Pilot '' 2005 -- 2006 Veronica Mars Gia Goodman Recurring role ; 8 episodes 2006 Bedford Diaries ,
</item 3>

<item 4>
Emmanuelle Vaugier
Stine 's The Haunting Hour Abigail Episode 1.06 : `` The Red Dress '' 2011 Killer Mountain Kate TV movie 2012 It 's Christmas , Carol ! Carol Huffman TV movie
2013 The Mentalist Melissa Enfield Episode 5.16 : `` There Will Be Blood '' 2013 Clara 's Deadly Secret Helen TV movie 2013 Hidden Away Alexandra Cole /
Stephanie Bennett TV movie 2015 My Stepdaughter Jill TV movie 2015 Mistresses Niko Recurring role ; 6 episodes 2016 Love In Paradise Heather Twain TV movie 2016
Stranger in the House Jade TV movie 2016 His Double Life Linda TV movie 2016 Love in the Vineyard ( Heart Felt ) Molly TV movie 2017 Rogue Regan Recurring role
; 7 episodes 2017 Washed Away Gabrielle Video Games ( edit ) Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying
Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice References ( edit ) Jump up ^ `` Archived copy '' .
</item 4>

<item 5>
Emmanuelle Vaugier
10 : `` Chivalry '' Madison Noella D'Angelo Recurring role ; 9 episodes The Halfback of Notre Dame Esmerelda TV movie Home Song Cheerleader TV movie The Limbic
Region Jennifer Lucca - age 21 TV movie 1997 Breaker High Monette Episode 1.11 : `` Chateau L'Feet J'mae '' 1997 Police Academy : The Series Sally Episode 1.03
`` Put Down That Nose '' 1997 Ninja Turtles : The Next Mutation Episode 1.12 : `` Turtles ' Night Out '' 1998 First Wave Esther Episode 1.07 : `` Lungfish ''
1998 Dead Man 's Gun Rose Harris Episode 2.10 : `` The Pinkerton '' 1998 -- 99 The Outer Limits Shal ; Lisa Dobkins Episode 4.08 : `` Rite of Passage '' ;
Episode 5.05 : `` The Other Side '' 1999 Viper Mitzi / Olga Episode 3.11 : `` Best Seller '' 1999 Assault on Death Mountain French Woman TV movie 1999 Seven
Days Princess Lisette D'Arcy Episode 2.
</item 5>

<item 6>
Emmanuelle Vaugier
She appeared in the film 40 Days and 40 Nights , which starred Josh Hartnett . She had starring roles in Suddenly Naked , in which she plays a Latin pop
sensation ; the TV - movie Mindstorm , a science - fiction thriller ; and Ripper , a psychological thriller about students who mysteriously disappear after
enrolling in a class about serial killers . She stars also in the sequel to Mirrors , which is directed by Victor Garcia . Maxim magazine featured her on the
cover of the February 2006 issue in the United States . Three months later , she landed spot # 31 on their annual Hot 100 list . Games ( edit ) Vaugier appeared
in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
</item 6>

<item 7>
Emmanuelle Vaugier
Alexandra `` Nightingale '' Morgan 2005 Saw II Addison Corday 2007 Unearthed Sheriff Annie Flynn 2007 Saw IV Addison Corday 2008 Blonde and Blonder Cat 2008
Bachelor Party 2 : The Last Temptation Eva Direct to video 2008 Far Cry Valerie Cardinal 2009 Dolan 's Cadillac Elizabeth Mirrors 2 Elizabeth Reigns Direct to
video Hysteria Jennifer A Nanny for Christmas Ally Leeds 2011 Where the Road Meets the Sun Lisa 2011 French Immersion Jennifer Yates 2012 Stolen Child Amanda
Direct to video 2013 Absolute Deception Rebecca 2013 Susie 's Hope Donna Smith Lawrence 2014 Teen Lust Shelley 2014 Saul : The Journey to Damascus Mary
Magdalene 2015 My Stepdaughter Jill 2016 Destruction : Los Angeles Margot Taylor Television ( edit ) Year Title Role Notes 1995 A Family Divided Rosalie Frank
TV movie 1995 Highlander : The Series Maria Alcobar Episode 4.10 : `` Chivalry '' Madison Noella D'Angelo Recurring role ;
</item 7>

<item 8>
The Divergent Series
Cast ( edit ) On March 11 , 2013 , it was announced that the roles of Tori , Christina , and Caleb had gone to Maggie Q , Zoë Kravitz , and Ansel Elgort ,
respectively . Ray Stevenson , Jai Courtney and Aaron Eckhart were announced to be in talks to join the cast on March 15 , 2013 , Stevenson and Courtney joined
the cast as Marcus Eaton and Eric , respectively . That same day , Miles Teller was cast as Peter . The studio 's announcement included a bit of casting news :
Mekhi Phifer will play Dauntless leader Max . Kate Winslet was announced to be in talks on January 24 , 2013 . Later it was confirmed that she would portray
Jeanine Matthews . Talking about playing the negative character for the first time , Winslet said , `` I 'm no idiot . The idea went through my head that I have
never played a baddie before , I was almost kind of surprised . '' Since Winslet joined the filming late , she used that distance from her co-stars to appear
aloof on the first day of her shoot .
</item 8>

<item 9>
Emmanuelle Vaugier
She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) . Filmography ( edit ) Film ( edit ) Year Title Role Notes 1998
Hysteria Veronica Bloom 1999 Shapeshifter Anika 1999 The Fear : Resurrection Jennifer 2000 My 5 Wives Sarah 2000 The Sculptress Sylvie 2001 Mindstorm Tracy
Wellman 2001 Wishmaster 3 : Beyond the Gates of Hell Elinor Smith Direct to video 2001 Ripper Andrea `` Andy '' Carter 2001 Suddenly Naked Lupe Martinez 2002 40
Days and 40 Nights Susie 2003 Secondhand Lions Jasmine 2003 Water 's Edge Rae Butler 2005 House of the Dead II Dr.
</item 9>

<item 10>
Sheila Carter
Snapper Foster Sam Gibson Diego Guittierez Nathan Hastings Jana Hawkes J.T. Hellstrom Juliet Helton Brittany Hodges Cole Howard Diane Jenkins Mamie Johnson
Leanna Love Ronan Malloy Dylan McAvoy Tucker McCall Ryan McNeil Tyler Michaelson Amber Moore Adam Newman Cassie Newman Cassandra Rawlins Gina Roma Daniel
Romalotti Danny Romalotti Luca Santori Deacon Sharpe Marisa Sierras Rex Sterling Heather Stevens Grace Turner Angelina Veneziano Ian Ward Sage Warner Nina
Webster Patty Williams Ricky Williams Hope Wilson Drucilla Winters Malcolm Winters Olivia Winters Character lists 1970s 1980s 1990s 2000s 2011 2012 2013 2014
2015 2016 2017 Newman family Abbott family Chancellor family Barber / Winters family The Bold and the Beautiful Current characters Maya Avant Nicole Avant
Sheila Carter Zende Dominguez Pamela Douglas Eric Forrester Ivy Forrester Rick Forrester Ridge Forrester R.J.
</item 10>
=== END ITEM id=a08ec3932a745a25 ===

=== ITEM id=c3b0b6f60a5229d5 ===
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
External links ( edit ) Wikiquote has quotations related to : Home Alone 2 : Lost in New York Home Alone 2 : Lost in New York on IMDb Home Alone 2 : Lost in New
York at the TCM Movie Database Home Alone 2 : Lost in New York at AllMovie Home Alone 2 : Lost in New York at Rotten Tomatoes Movie Locations Guide.com - Maps
and directions to Home Alone 2 Filming Locations Home Alone Films Home Alone ( 1990 ) Home Alone 2 : Lost in New York ( 1992 ) Home Alone 3 ( 1997 ) Home Alone
4 ( 2002 ) Home Alone : The Holiday Heist ( 2012 ) Video games Home Alone ( 1991 ) Home Alone 2 : Lost in New York ( 1992 ) Home Alone ( 2006 ) Miscellaneous
Home Alone : Original Motion Picture Soundtrack Talkboy Chris Columbus Directed Adventures in Babysitting ( 1987 ) Heartbreak Hotel ( 1988 ) Home Alone ( 1990 )
Only the Lonely ( 1991 ) Home Alone 2 : Lost in New York ( 1992 ) Mrs.
</item 1>

<item 2>
Home Alone 2: Lost in New York
Contents ( hide ) 1 Plot 2 Cast 2.1 Main cast 2.2 The McCallister family 2.3 Other characters 3 Release 3.1 Marketing 3.2 Box office 3.3 Critical reception 3.4
Home media 3.5 Social Media 4 Music 4.1 Original Score 4.2 Original Soundtrack Album 4.3 The Deluxe Edition 5 Novelization 6 Sequels 7 See also 8 References 9
External links Plot ( edit ) In Chicago , the McCallister family is preparing for a Christmas vacation in Miami . On the night before their departure , the
entire family gathers at Peter and Kate 's home , where their youngest son , Kevin , sees Florida as contradictory to Christmas , on his opinion of the lack of
Christmas trees in Florida . During the school Christmas pageant , Kevin 's older brother Buzz humiliates him during his solo , causing Kevin to retaliate ,
thereby ruining the performance .
</item 2>

<item 3>
Home Alone 2: Lost in New York
Duncan . Kevin learns that the proceeds from the store 's Christmas sales will be donated to a children 's hospital . Kevin donates $20 to the hospital . As a
token of appreciation , Mr. Duncan offers Kevin a pair of ceramic turtledoves as a gift , instructing him to give one to another person as a sign of eternal
friendship . After running into Harry and Marv , who have recently escaped from prison and are now called `` The Sticky Bandits '' , Kevin retreats to the Plaza
Hotel . The hotel 's concierge , Mr. Hector , confronts Kevin about the credit card , which has been reported stolen . Kevin flees after evading Mr. Hector ,
but is captured by Harry and Marv . The duo discuss plans for breaking into the toy store that night . Kevin manages to escape by pinching a woman in front of
him , which results in the woman attacking the duo . Kevin 's family travels to New York after tracking the whereabouts of the stolen credit card , and Kate
searches the city for Kevin .
</item 3>

<item 4>
Home Alone 2: Lost in New York
When the duo chase Kevin around the townhouse , he escapes and calls the police . Harry and Marv catch him and take him down to Central Park to kill him , but
the pigeon lady sneaks in and throws birdseed on the duo , and the pigeons storm in to incapacitate them . Kevin sets off fireworks he had bought earlier to
signal the police . The police arrive and arrest the bandits . At the toy store , Mr. Duncan finds a note from Kevin attached to the brick , explaining about
what happened . Kate , still searching for Kevin on foot , remembers his fondness for Christmas trees . After observing Kevin making a wish at the Rockefeller
Center Christmas Tree , Kate meets him there , and they reconcile . On Christmas Day , a truckload of gifts arrive at the McCallisters ' hotel room from the toy
store . Kevin and Buzz reconcile , and Buzz allows him to open up the first present . Kevin goes to Central Park to give the pigeon lady the second turtledove .
At the Plaza , Buzz receives the bill for Kevin 's stay from Cedric and shows it to Peter .
</item 4>

<item 5>
Home Alone 2: Lost in New York
It is the second film in the Home Alone series and the sequel to Home Alone . Macaulay Culkin reprises his role as Kevin McCallister , while Joe Pesci and
Daniel Stern reprise their roles as the Wet Bandits , now known as the Sticky Bandits . Catherine O'Hara , John Heard , Rob Schneider , Tim Curry , and Brenda
Fricker are also featured . The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New
York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet
Bandits and must outrun and out - prank them again . The film was shot in Winnetka , Illinois ; O'Hare International Airport in Chicago ; Evanston , Illinois ;
and New York City ( which was star Culkin 's hometown at the time ) . The exterior of Duncan 's Toy Chest in New York City was filmed outside of the Rookery
Building in downtown Chicago .
</item 5>

<item 6>
Home Alone 2: Lost in New York
In the beginning of the novelization , a prologue , which ends up being Marv 's nightmare in prison , he and Harry sneak away from the cops and return to Kevin
's house to seek revenge on Kevin . Kevin bolts into the garage with Marv and Harry in hot pursuit . Harry and Marv end up triggering extra traps that Kevin had
set up in the garage . Kevin watches as Marv ends up triggering a trap where a running lawnmower falls on his head ( This was a trap featured in Home Alone 3 )
. Sequels ( edit ) Main article : Home Alone 3 A second sequel , Home Alone 3 , followed in 1997 . Two additional television movie sequels , Taking Back the
House and The Holiday Heist , would follow in 2002 and 2012 , respectively . See also ( edit ) List of films featuring fictional films References ( edit ) Jump
up ^ `` HOME ALONE 2 - LOST IN NEW YORK ( PG ) '' . British Board of Film Classification . November 4 , 2017 . Retrieved November 10 , 2015 .
</item 6>

<item 7>
Home Alone 2: Lost in New York
While Buzz fakes an apology to Kevin in front of the McCallisters ( who are convinced it is sincere ) , Kevin refuses to apologize to Buzz and storms to the
third floor of the house . Prior to the pageant , Peter has unknowingly caused the alarm clock to reset ; consequently , the family once again oversleeps . In
the confusion they have to rush to reach the airport on time , Kevin falls behind everyone else , and mistakenly boards a flight bound for New York City while
trying to replace the batteries for a Talkboy , carrying Peter 's bag containing his wallet and a large amount of money ; upon arrival in Miami , Kate realizes
that Kevin is missing again . In New York , Kevin tours the city and meets a homeless woman tending to pigeons at Central Park , where he runs off . Kevin goes
to the Plaza Hotel , where he tantalizes the staff into checking in with Peter 's credit card . On Christmas Eve , Kevin tours the city in a limousine and
visits a toy store where he meets its philanthropic owner , Mr. Duncan .
</item 7>

<item 8>
Home Alone 2: Lost in New York
Meanwhile , Kevin goes to his uncle Rob 's townhouse , only to find the house vacant and undergoing renovations , while Rob and his family are in Paris . At
Central Park , he encounters and befriends the pigeon lady . They go to Carnegie Hall , where an orchestra is performing . The pigeon lady explains how her life
collapsed and how she dealt with it by taking care of the pigeons in the park . Kevin gives the pigeon lady some advice and promises that he will be her friend
. Kevin , after remembering what the bandits said , returns to the townhouse , and rigs it with numerous booby traps . Kevin arrives at the toy store during
Harry and Marv 's robbery and throws a brick through the window , setting off the store 's alarm . Kevin then lures the duo to the townhouse , where he springs
the traps and Harry and Marv suffer various injuries . When the duo chase Kevin around the townhouse , he escapes and calls the police .
</item 8>

<item 9>
Home Alone 2: Lost in New York
Peter suddenly calls out , `` Kevin , you spent $967 on room service ? ! '' at which point Kevin runs off . Cast ( edit ) Main cast ( edit ) Macaulay Culkin as
Kevin McCallister Joe Pesci as Harry Lime Daniel Stern as Marv Merchants The McCallister family ( edit ) Catherine O'Hara as Kate McCallister John Heard as
Peter McCallister Devin Ratray as Buzz McCallister , Kevin 's brother who gets him into trouble Kieran Culkin ( Macaulay Culkin 's younger brother ) as Fuller
McCallister Gerry Bamman as Frank McCallister , Kevin 's uncle Hillary Wolf as Megan McCallister Maureen Elisabeth Shay ( who replaces Angela Goethals ) as
Linnie McCallister Mike Maronna as Jeff McCallister Terrie Snell as Leslie McCallister , Kevin 's aunt Jedidiah Cohen as Rod McCallister Senta Moses as Tracy
McCallister Daiana Campeanu as Sondra McCallister Anna Slotky as Brooke McCallister Other characters ( edit ) Brenda Fricker as the Pigeon Lady Tim Curry as Mr.
</item 9>

<item 10>
Home Alone 2: Lost in New York
Disc One Home Alone ( Main Title ) ( 2 : 07 ) This Year 's Wish ( 1 : 47 ) We Overslept Again / Holiday Flight ( 3 : 19 ) Separate Vacations * ( 1 : 58 )
Arrival in New York * * ( 2 : 59 ) The Thieves Return ( 3 : 28 ) Plaza Hotel ( 3 : 04 ) Concierge ( 1 : 31 ) Distant Goodnights ( Christmas Star ) ( Lyrics by
Leslie Bricusse ) ( 2 : 05 ) A Day in the City ( : 59 ) Duncan 's Toy Store ( 2 : 41 ) Turtle Doves ( 1 : 29 ) To the Plaza , Presto ( 3 : 27 ) Race to the Room
/ Hot Pursuit ( 4 : 08 ) Haunted Brownstone ( 3 : 02 ) Appearance of the Pigeon Lady ( 3 : 21 ) Christmas at Carnegie Hall ( 5 : 15 ) O Come ,
</item 10>
=== END ITEM id=c3b0b6f60a5229d5 ===
