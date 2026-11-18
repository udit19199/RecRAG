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

=== ITEM id=abb0892b2920300b ===
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
List of narrative techniques
Author surrogate Characters which are based on authors , usually to support their personal views . Sometimes an intentionally or unintentionally idealized
version of them . A variation is the Mary Sue or Gary Stu , which primarily serves as an idealized self - insertion . Socrates in the writings of Plato . Plato
never speaks in his own voice in his dialogues . In the Second Letter , it says , `` no writing of Plato exists or ever will exist , but those now said to be
his are those of a Socrates become beautiful and new '' . Breaking the fourth wall An author or character addresses the audience directly ( also known as direct
address ) . This may acknowledge to the reader or audience that what is being presented is fiction , or may seek to extend the world of the story to provide the
illusion that they are included in it . The characters in Sesame Street often break the fourth wall when they address their viewers as part of the ongoing
storyline , which is possible because of the high level of suspension of belief afforded by its audience -- children . The American political drama show House
of Cards also uses this technique frequently to let the viewers know what the main character Frank Underwood is thinking and planning .
</item 1>

<item 2>
List of narrative techniques
As a character , she is telling the `` 1,001 stories '' to the King , in order to delay her execution night by night . However , as a framing device her purpose
for existing is to tell the same 1,001 stories to the reader . MacGuffin A plot device coined by Alfred Hitchcock referring to some goal , desired object , or
other motivator that the protagonist pursues , often with little or no narrative explanation as to why it is considered so important . In Hitchcock 's The 39
Steps the MacGuffin is a mysterious set of `` military secrets . '' No one knows what they are , and in the end they mean almost nothing . For all its power ,
mystery , and danger , The One Ring in Lord of the Rings is a MacGuffin that saves the world . The stuff that dreams are made of . The Maltese Falcon is a
powerful MacGuffin in the film of the same name , a supposedly jewel encrusted black bird which creates the greed which propels every character , even the hero
. In medias res Beginning the story in the middle of a sequence of events .
</item 2>

<item 3>
List of narrative techniques
The Master and Margarita uses an omniscient narrator . Style ( edit ) See also : Figure of speech Name Definition Example Allegory A symbolic fiction story .
C.S. Lewis 's The Lion , the Witch , and the Wardrobe is a religious allegory with Aslan as Christ and Edmund as Judas . Alliteration Repeating the same letter
or consonant sound at the beginning of adjacent or closely connected words . In the film V for Vendetta the main character performs a couple of soliloquies with
a heavy use of alliteration , e.g. , `` Voilà ! In view , a humble vaudevillian veteran , cast vicariously as both victim and villain by the vicissitudes of
Fate . This visage , no mere veneer of vanity , is a vestige of the vox populi , now vacant , vanished , as the once vital voice of the verisimilitude now
venerates what they once vilified .
</item 3>

<item 4>
List of narrative techniques
Defamiliarization Taking an everyday object and presenting it in a way that is weirdly unfamiliar so that we see the object in a new way . Coined by the early
20th - century Russian literary critic Viktor Shklovsky in `` Art as Technique . '' In , Swift 's Gulliver 's Travels when Gulliver visits the land of the
giants and sees a giant woman 's skin he sees it is anything but smooth and beautiful when viewed up close . First - person narration A text presented from the
point of view of a character , especially the protagonist , as if the character is telling the story themselves . ( Breaking the fourth wall is an option , but
not a necessity , of this format . ) Mark Twain 's Adventures of Huckleberry Finn uses the title character as the narrator , while Sherlock Holmes is primarily
told from Watson 's perspective . Magical realism Describing events in a real - world setting but with magical trappings , often incorporating local customs and
invented beliefs . Different from urban fantasy in that the magic itself is not the focus of the story . Particularly popular with Latin American authors like
Gabriel García Márquez and Jorge Luis Borges .
</item 4>

<item 5>
List of narrative techniques
If it 's not going to be fired , it should n't be hanging there . '' -- Anton Chekhov Unreliable narrator The narrator of the story is not sincere , or
introduces a bias in their narration and possibly misleads the reader , hiding or minimizing events , characters , or motivations . An example is The Murder of
Roger Ackroyd . The novel includes an unexpected plot twist at the end of the novel . In the last chapter , Sheppard describes how he was an unreliable narrator
. Perspective ( edit ) Name Definition Example Audience surrogate A character who expresses the questions and confusion of the audience , with whom the audience
can identify . Frequently used in detective fiction and science fiction , where the character asks a central character how he or she accomplished certain deeds
, for the purpose of inciting that character to explain ( for the curious audience ) his or her methods , or a character asking a relatively educated person to
explain what amounts to the backstory . Dr Watson in the Sherlock Holmes stories . Scott Evil , played by Seth Green , son of Dr. Evil on the Austin Powers
movies . Author surrogate Characters which are based on authors , usually to support their personal views .
</item 5>

<item 6>
List of narrative techniques
Her tears were a river flowing down her cheeks . Thematic patterning Distributing recurrent thematic concepts and moralistic motifs among various incidents and
frames of a story . In a skillfully crafted tale , thematic patterning may emphasize the unifying argument or salient idea disparate events and disparate frames
have in common . Each of the chapters of Ulysses by James Joyce . Character ( edit ) Name Type Notes Anthropomorphism Form of personification that applies human
- like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis
Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw .
Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in
the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors
Lear 's mental deterioration .
</item 6>

<item 7>
List of narrative techniques
His parents died protecting him , and when Voldemort tried to cast a killing curse on Harry , it rebounded and took away most of his strength , and gave Harry
Potter a unique ability and connection with the Dark Lord thus marking him as his equal . Story within a story ( Hypodiegesis ) A story told within another
story . See also frame story . In Stephen King 's The Wind Through the Keyhole , of the Dark Tower series , the protagonist tells a story from his past to his
companions , and in this story he tells another relatively unrelated story . Ticking clock scenario Threat of impending disaster -- often used in thrillers
where salvation and escape are essential elements In the TV show `` 24 '' , the main character , Jack Bauer often finds himself interrogating a terrorist who is
caught in order to disarm a bomb . Chekhov 's gun A dramatic principle that requires every element in a narrative to be irreplaceable , with anything else
removed . `` Remove everything that has no relevance to the story . If you say in the first chapter that there is a rifle hanging on the wall , in the second or
third chapter it absolutely must go off .
</item 7>

<item 8>
Atticus Finch
This kind of character development , where motivations and ideals between characters , for reasons of plot , are changed is not unusual in the process of
creative writing . Apart from the more progressive depiction of Atticus , the depiction of the town itself , especially the African - American characters , is
also dramatically altered between the two drafts. Real - life comparisons with Lee 's father , Amasa Coleman Lee , have also been made in the two differing
versions of Atticus in that originally Amasa Lee was in favor of segregation but became more liberal later in life , later changing his views to those of
Integration . Tay Hohoff , Lee 's editor , has also been argued to have played a major part in the character development of the novel and particularly Atticus '
liberal transformation . Jonathan Mahler of The New York Times notes in his article The Invisible Hand Behind Harper Lee 's ' To Kill a Mockingbird ' that Ms.
Hohoff , at the same time as she was guiding Ms. Lee through the Mockingbird re-write , was working on her own biography of the early - 20th - century New York
activist and humanist John Lovejoy Elliot .
</item 8>

<item 9>
List of narrative techniques
Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II
of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . '' References ( edit ) Jump up ^ Orehovec , Barbara ( 2003 ) .
Revisiting the Reading Workshop : A Complete Guide to Organizing and Managing an Effective Reading Workshop That Builds Independent , Strategic Readers (
illustrated ed . ) . Scholastic Inc. p. 89 . ISBN 0439444047 . Jump up ^ Demchick , Harrison ( 2013 - 09 - 26 ) . `` Techniques and Tension in Breaking Bad '' .
Huffington Post . Retrieved 2017 - 11 - 14 . Jump up ^ `` Literature Glossary - Defamiliarization '' . www.shmoop.com . Retrieved 2017 - 11 - 14 . Jump up ^ ``
Allegory Examples '' . YourDictionary . Retrieved 2017 - 11 - 14 .
</item 9>

<item 10>
List of narrative techniques
29 . ISBN 1 - 134 - 50341 - 5 . Jump up ^ Heath , Peter ( May 1994 ) , `` Reviewed work ( s ) : Story - Telling Techniques in the Arabian Nights by David
Pinault '' , International Journal of Middle East Studies , Cambridge University Press , 26 ( 2 ) : 358 -- 360 , doi : 10.1017 / s0020743800060633 Jump up ^
Heath ( 1994 ) p. 360 Jump up ^ `` Personification - Examples and Definition of Personification '' . 2013 - 06 - 27 . Retrieved 2016 - 08 - 16 . Narrative
Character Antagonist Antihero Archenemy Character arc Characterization Deuteragonist False protagonist Focal character Foil Protagonist Stock character
Supporting character Tritagonist Narrator Tragic hero Plot Action Backstory Cliché Climax Cliffhanger Conflict Deus ex machina Dialogue Dramatic structure
Exposition Eucatastrophe Foreshadowing Flashback Flashforward Frame story In medias res Pace Plot device Plot twist Poetic justice Reveal Self - fulfilling
prophecy Subplot Trope
</item 10>
=== END ITEM id=abb0892b2920300b ===

=== ITEM id=50841ea20db87a44 ===
Gold sentences and facts:
Sentence 0: The nineteenth season of Law & Order : Special Victims Unit premiered on September 27 , 2017 .
  Fact 0.0: The nineteenth season of Law & Order: Special Victims Unit premiered.
  Fact 0.1: The nineteenth season of Law & Order: Special Victims Unit premiered on September 27, 2017.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 1>

<item 2>
Law & Order: Special Victims Unit
ET , where it stayed until the twelfth season . In the 12th season , SVU moved back to 9 : 00 p.m. to lead in the newest Law & Order spinoff , Law & Order : LA
, until it was pulled from the network in January 2011 to be retooled . SVU moved back to 10 : 00 p.m. on January 12 , 2011 , until the end of the 13th season .
With season 14 , SVU moved back to 9 : 00 p.m. after a two - hour season premiere event on September 26 , 2012 , a position the show has held since . Russian
adaptation ( edit ) In 2007 , the Russian production company Studio 2B purchased the rights to create an adaptation of Law & Order : Special Victims Unit for
Russian television . Titled Закон и порядок : отдел оперативных расследований ( Law & Order : Division of Field Investigation ) , the series stars Alisa Bogart
as Major Olga Bobrova .
</item 2>

<item 3>
Law & Order: Special Victims Unit
2017 ( 2017 - 09 - 27 ) TBA TBA TBA Reception ( edit ) U.S. television Ratings ( edit ) Law & Order : Special Victims Unit premiered on Monday , September 20 ,
1999 . After nine episodes , the show was moved to Friday nights , where it found its audience and following its first and second seasons became a Top 20 -
overall show , overcoming the `` Friday night death slot '' phenomenon . Beginning with the season five , the show aired on Tuesdays to compete with CBS '
Judging Amy and ABC 's NYPD Blue . In its later years , Law & Order : Special Victims Unit consistently outperformed Law & Order in the Nielsen ratings for
first run episodes until the latter 's cancellation in 2010 . In 2016 , a New York Times study of the 50 TV shows with the most Facebook Likes found that SVU 's
popularity was `` atypical : generally slightly more popular in rural areas and the Black Belt , but largely restricted to the eastern half of the country . It
's most popular in Albany , N.Y. ; least in Colorado and Utah '' .
</item 3>

<item 4>
Law & Order: Special Victims Unit
The show also briefly ran on Syfy in 2006 . International broadcast ( edit ) In Australia , Law & Order : SVU airs on Network Ten and Australia 's cable channel
Universal Channel ( formerly aired on TV1 which ceased to broadcast in December 2013 ) . In the UK `` Law & Order : SVU '' originally aired on the Hallmark
Channel ( which in 2012 became the Universal Channel ) with a terrestrial TV airing on Channel 5 in 2016 which then switched to Five USA . The UK is two and
half weeks behind the US . In Brazil , Law & Order : SVU also airs on Universal Channel , with weekdays episodes from past seasons and simultaneously , two
episodes after the original airing in the US . Law & Order : SVU is broadcast at 9.30 pm on Tuesdays on TV3 in New Zealand . Series 18 premiered on 11 April
2017 . References ( edit ) Notes ( edit ) Jump up ^ SVU Writers Room ( @ SVUWritersRoom ) ( 7 July 2016 ) . `` First cast read - through of # SVUSeason18 '' (
Tweet ) -- via Twitter .
</item 4>

<item 5>
Law & Order: Special Victims Unit (season 19)
Missing or empty title = ( help ) ( hide ) Law & Order : Special Victims Unit Characters Detectives Elliot Stabler Olivia Benson John Munch Monique Jeffries Fin
Tutuola Chester Lake Nick Amaro Amanda Rollins Dominick Carisi Jr . Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot
Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya
Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' ``
Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion
'' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion '' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and
nominations Law & Order franchise Retrieved from `` https://en.
</item 5>

<item 6>
Law & Order: Special Victims Unit
Gillian Hardwicke served as the SVU ADA during season twelve . In season thirteen , both Cabot and Novak returned as ADAs and since the beginning of season
fourteen , ADA Rafael Barba has been SVU 's prosecutor . In season 13 other big changes happened when Stabler left in the season twelve finale . Detectives Nick
Amaro and Amanda Rollins joined the team filling the void left by Stabler . Amaro brings empathy to his cases while dealing with a stressful home life , whilst
Rollins ' dogged persistence and instincts help her close cases , but her secrets could derail her career . In the current seasons the Manhattan SVU is now run
by Lieutenant Olivia Benson . Episodes ( edit ) Main article : List of Law & Order : Special Victims Unit episodes Law & Order : Special Victims Unit has
concluded its eighteenth season and has started its nineteenth season . Each season has aired on NBC and consists of 19 to 25 episodes , each lasting
approximately forty minutes ( sixty minutes including commercials ) . Season Episodes Originally aired Nielsen ratings First aired Last aired Rank Rating 22
September 20 ,
</item 6>

<item 7>
Law & Order: Special Victims Unit
ET for the first nine episodes , from September 20 through November 29 , 1999 . It was then shifted to Friday nights at 10 p.m. ET on January 7 , 2000 , and
remained in that time slot through the end of season four on May 16 , 2003 . Beginning with the season five premiere on September 23 , 2003 , SVU moved to
Tuesday nights at 10 : 00 p.m. ET . NBC airs repeats on Saturday nights at varying times and previous episodes are shown on the USA Network on varying days in
marathon blocks With the season eleven premiere on September 23 , 2009 , the series vacated its Tuesday 10 p.m. ET slot because NBC began a prime - time
weeknight Jay Leno series . The new time slot became Wednesday nights at 9 : 00 p.m. ET on NBC , with CTV still airing SVU on Tuesdays at 10 : 00 in Canada .
After the 2010 Winter Olympics on March 3 , 2010 , the time slot for SVU changed again to Wednesdays at 10 p.m.
</item 7>

<item 8>
Law & Order: Special Victims Unit
The first episode , `` Payback '' , premiered on NBC on September 20 , 1999 . Executive producers ( edit ) Executive producer Neal Baer left Law & Order : SVU
as showrunner at the end of season twelve , after eleven years ( seasons 2 -- 12 ) on the show , in order to sign a three - year deal with CBS Studios . Baer
was replaced by former Law & Order : Criminal Intent showrunner Warren Leight . In March 2015 , it was announced that Warren Leight signed a three - year deal
with Sony Pictures Television , that will allow him to work on SVU one more season , its seventeenth . Leight joined the show in season thirteen . It was
announced on March 10 , 2016 that original Law & Order veteran producer Rick Eid would take Leight 's place as showrunner starting in season 18 . Creator Dick
Wolf commented to The Hollywood Reporter , `` I 'm extremely pleased that Rick had decided to rejoin the family and hope that he will be here for years to come
. ''
</item 8>

<item 9>
Law & Order: Special Victims Unit
During post-production of season 18 , following the announcement that SVU was renewed for a nineteenth season , it was revealed that Rick Eid departed the
series . He will be taking over another Dick Wolf / NBC series . Chicago P.D. It was announced on May 25 , 2017 that original Law & Order and Law & Order :
Criminal Intent showrunner Michael S. Chernuchin would be reprising his role starting on season nineteen . Chernuchin was also co-creator and executive
producing showrunner of Chicago Justice , another Wolf - related show that was canceled by NBC at the end of the 2016 -- 17 TV season . Filming ( edit ) SVU
shooting on location in Central Park at night Many exterior scenes of Law & Order : Special Victims Unit are filmed on location in New York City , Wolf 's
hometown , throughout all five of New York City 's boroughs . As the NYPD encounters varied law enforcement challenges on a daily basis , the backdrop provides
the writers a supply of ideal locations from which to choose . Fort Lee , New Jersey served as the filming location for Detective Elliot Stabler 's residence in
Queens , New York .
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
=== END ITEM id=50841ea20db87a44 ===

=== ITEM id=77ae4344f2c3bbe9 ===
Gold sentences and facts:
Sentence 0: The book became an instant hit , and the online version received about 20 million views as of 2009 ; many online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
  Fact 0.0: The book became an instant hit.
  Fact 0.1: The online version of the book received about 20 million views as of 2009.
  Fact 0.2: Many online readers requested a printed version of the book.
  Fact 0.3: Kinney agreed to a printed version of the book.
  Fact 0.4: Diary of a Wimpy Kid was published in April 2007.
  Fact 0.5: To date, twelve Wimpy Kid books have been released.
  Fact 0.6: To date, one do-it-yourself Wimpy Kid book has been released.
  Fact 0.7: To date, two Wimpy Kid movie diaries have been released.
  Fact 0.8: In April 2009, TIME named Kinney one of The World's Most Influential People.
Sentence 1: # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 - 0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer . The Last Straw January 13 , 2009 ISBN 978 - 0 - 8109 - 7068 - 7 When Greg 's father threatens to send Greg to a military academy , Greg must find a way to prevent this from happening . Dog Days October 12 , 2009 ISBN 978 - 0 - 8109 - 8391 - 5 Summer is here , but when the family has not that much money to go to the beach ( money is tight at that time ) , it might not be the best summer ever . 5 The Ugly Truth November 9 , 2010 ISBN 978 - 0 - 8109 - 8491 - 2 Greg is growing up , and he will have to face the challenges of this period ( like changing dentists , and a school lock - in ) . 6 Cabin Fever November 15 , 2011 ISBN 978 - 1 - 4197 - 0223 - 5 The family is stranded in the house as snow begins to fall . 7 The Third Wheel November 13 , 2012 ISBN 978 - 1 - 4197 - 0584 - 7 The Valentine 's Day dance is coming up , and Greg has no one go with . But when Michael Sampson ( Abigail 's date ) will not be going , things turn around . 8 Hard Luck November 5 , 2013 ISBN 978 - 1 - 4197 - 1132 - 9 Greg 's friend count is dropping as Rowley now has a girlfriend , named Abigail ( continuing the last book ) . 9 The Long Haul November 4 , 2014 ISBN 978 - 1 - 4197 - 1189 - 3 This is the family 's second summer trip , but instead , it 's a road trip . But with the Heffleys , you know what to expect from this disaster . 10 Old School November 3 , 2015 ISBN 978 - 1 - 4197 - 1701 - 7 The neighborhood is going to cut electricity for the weekend , and as the family 's car has a fender bender , it forces Greg to go to the farm trip . Will Greg survive the electricity - free days ? 11 Double Down November 1 , 2016 ISBN 978 - 1 - 4197 - 2344 - 5 Greg 's mom wants him to do something else than video games , so he makes a movie in collaboration with Rowley and joins the music program . 12 The Getaway November 7 , 2017 ISBN 978 - 1 - 4197 - 2545 - 6 Instead of celebrating the holidays at home , Greg and his family head on a tropical vacation to a resort . But what first starts as an relaxing trip , later turns into a Christmas nightmare . Will Greg be able to turn this vacation around ?
  Fact 1.0: Diary of a Wimpy Kid was released on April 1, 2007, with ISBN 978-0-8109-9313-6.
  Fact 1.1: The introduction to the Diary of a Wimpy Kid series introduces the protagonist Greg Heffley, his best friend Rowley, his middle school problems, Halloween, and the Cheese Touch.
  Fact 1.2: Rodrick Rules was released on February 1, 2008, with ISBN 978-0-8109-9473-7.
  Fact 1.3: In Rodrick Rules, Greg must stop Rodrick from revealing the secrets that happened last summer.
  Fact 1.4: The Last Straw was released on January 13, 2009, with ISBN 978-0-8109-7068-7.
  Fact 1.5: In The Last Straw, when Greg's father threatens to send Greg to a military academy, Greg must find a way to prevent this.
  Fact 1.6: Dog Days was released on October 12, 2009, with ISBN 978-0-8109-8391-5.
  Fact 1.7: In Dog Days, summer is here, but the family does not have much money to go to the beach, so it might not be the best summer ever.
  Fact 1.8: The Ugly Truth was released on November 9, 2010, with ISBN 978-0-8109-8491-2.
  Fact 1.9: In The Ugly Truth, Greg is growing up and will have to face the challenges of this period, like changing dentists and a school lock-in.
  Fact 1.10: Cabin Fever was released on November 15, 2011, with ISBN 978-1-4197-0223-5.
  Fact 1.11: In Cabin Fever, the family is stranded in the house as snow begins to fall.
  Fact 1.12: The Third Wheel was released on November 13, 2012, with ISBN 978-1-4197-0584-7.
  Fact 1.13: In The Third Wheel, the Valentine's Day dance is coming up and Greg has no one to go with.
  Fact 1.14: In The Third Wheel, things turn around when Michael Sampson, Abigail's date, will not be going.
  Fact 1.15: Hard Luck was released on November 5, 2013, with ISBN 978-1-4197-1132-9.
  Fact 1.16: In Hard Luck, Greg's friend count is dropping as Rowley now has a girlfriend, named Abigail.
  Fact 1.17: Hard Luck continues the story from The Third Wheel.
  Fact 1.18: The Long Haul was released on November 4, 2014, with ISBN 978-1-4197-1189-3.
  Fact 1.19: In The Long Haul, this is the family's second summer trip, and it is a road trip.
  Fact 1.20: Old School was released on November 3, 2015, with ISBN 978-1-4197-1701-7.
  Fact 1.21: In Old School, the neighborhood is going to cut electricity for the weekend.
  Fact 1.22: In Old School, the family's car has a fender bender, which forces Greg to go on the farm trip.
  Fact 1.23: Old School asks whether Greg will survive the electricity-free days.
  Fact 1.24: Double Down was released on November 1, 2016, with ISBN 978-1-4197-2344-5.
  Fact 1.25: In Double Down, Greg's mom wants him to do something other than video games, so he makes a movie in collaboration with Rowley.
  Fact 1.26: In Double Down, Greg joins the music program.
  Fact 1.27: The Getaway was released on November 7, 2017, with ISBN 978-1-4197-2545-6.
  Fact 1.28: In The Getaway, instead of celebrating the holidays at home, Greg and his family head on a tropical vacation to a resort.
  Fact 1.29: In The Getaway, what first starts as a relaxing trip later turns into a Christmas nightmare.
  Fact 1.30: The Getaway asks whether Greg will be able to turn the vacation around.
Sentence 2: Diary of a Wimpy Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films . Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback ) Audiobook E-book
  Fact 2.0: The logo of the Diary of a Wimpy Kid series appears on the book covers, blogs, web comic, and films.
  Fact 2.1: The Diary of a Wimpy Kid series includes the books Rodrick Rules, The Last Straw, Dog Days, The Ugly Truth, Cabin Fever, The Third Wheel, Hard Luck, The Long Haul, Old School, Double Down, and The Getaway.
  Fact 2.2: The Diary of a Wimpy Kid series was written by Jeff Kinney.
  Fact 2.3: The Diary of a Wimpy Kid series is from the United States.
  Fact 2.4: The Diary of a Wimpy Kid series is in English.
  Fact 2.5: The Diary of a Wimpy Kid series is in the realistic fiction genre.
  Fact 2.6: The Diary of a Wimpy Kid series is published by Scholastic Publishing, Amulet Books, and Puffin Books.
  Fact 2.7: The Diary of a Wimpy Kid series was published from April 1, 2007 to the present.
  Fact 2.8: The Diary of a Wimpy Kid series is available in print (hardback and paperback), as an audiobook, and as an e-book.

Retrieved items:
<item 1>
Diary of a Wimpy Kid (book series)
Will Greg be able to turn this vacation around ? Supplementary books ( edit ) Diary of a Wimpy Kid Do - It - Yourself Book ( published between Rodrick Rules and
The Last Straw ) is an activity book that features a blank section encouraging readers to keep their own journal in the style of the books . The Wimpy Kid Do -
It - Yourself Book : Extended Edition ( published after Wimpy Kid Movie Diary 2 and before Cabin Fever ) is the same book as the first Do - It - Yourself Book ,
but with extra 60 pages and 16 more full - color comics . The Wimpy Kid Movie Diary is a book about the making of the first film , which features stills and
brand - new illustrations . Jeff Kinney dedicated the book to the film 's main stars , Zachary Gordon and Robert Capron . The Wimpy Kid Movie Diary 2 : Rodrick
Rules is an updated version of the first book , now featuring stills from the second film Rodrick Rules , more new illustrations , and information on how the
second film was made .
</item 1>

<item 2>
Diary of a Wimpy Kid (book series)
7 The Third Wheel November 13 , 2012 ISBN 978 - 1 - 4197 - 0584 - 7 The Valentine 's Day dance is coming up , and Greg has no one go with . But when Michael
Sampson ( Abigail 's date ) will not be going , things turn around . 8 Hard Luck November 5 , 2013 ISBN 978 - 1 - 4197 - 1132 - 9 Greg 's friend count is
dropping as Rowley now has a girlfriend , named Abigail ( continuing the last book ) . 9 The Long Haul November 4 , 2014 ISBN 978 - 1 - 4197 - 1189 - 3 This is
the family 's second summer trip , but instead , it 's a road trip . But with the Heffleys , you know what to expect from this disaster .
</item 2>

<item 3>
Diary of a Wimpy Kid (book series)
All the main books are the journals of the main character , Greg Heffley . Befitting a teen 's diary , the books are filled with hand - written notes and simple
drawings of Greg 's daily adventures . Since the release of the online version in May 2004 , most of the books have garnered positive reviews and commercial
success . As of February 2008 , it has been purchased one million times . The first , second , fourth , and ninth installments have been adapted into films by
20th Century Fox . The website Poptropica released islands in the theme of Diary of a Wimpy Kid called Wimpy Wonderland and Wimpy Boardwalk , both of which
credit Kinney in an ambiguous contributor 's role .
</item 3>

<item 4>
Diary of a Wimpy Kid (book series)
, to turn Diary of a Wimpy Kid into a print series . The book became an instant hit , and the online version received about 20 million views as of 2009 ; many
online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have
been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
Chronology ( edit ) # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the
protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 -
0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer .
</item 4>

<item 5>
Diary of a Wimpy Kid (book series)
Contents ( hide ) 1 Background 2 Chronology 2.1 Supplementary books 2.2 Toys and games 2.3 Audiobooks 3 Characters 4 Achievements 4.1 Awards and honors 4.2
Commercial success 5 Films 6 Musical 7 References 8 External links Background ( edit ) In January 1998 , Diary of a Wimpy Kid officially began , when Kinney
brought up an idea about a middle - school weakling named Greg Heffley , who wrote about his personal life . In May 2004 , FunBrain and Kinney released an
online version of Diary of a Wimpy Kid . The website made daily entries until June 2005 . He worked on his book for almost eight years , before showing it to a
publisher in New York . In February 2006 , during the New York Comic Con , Jeff signed a multi book deal with publisher Harry N. Abrams , Inc. , to turn Diary
of a Wimpy Kid into a print series .
</item 5>

<item 6>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly
Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a
Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre
Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback )
Audiobook E-book Diary of a Wimpy Kid is a series of fiction books written by the American author and cartoonist Jeff Kinney . All the main books are the
journals of the main character , Greg Heffley .
</item 6>

<item 7>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid ( book series ) - Wikipedia Diary of a Wimpy Kid ( book series ) This is the latest accepted revision , reviewed on 22 April 2018 . Jump to
: navigation , search `` Diary of a Wimpy Kid ( series ) '' redirects here . For the film series , see Diary of a Wimpy Kid ( film series ) . Diary of a Wimpy
Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films .
</item 7>

<item 8>
Diary of a Wimpy Kid (book series)
2007 ) Diary of a Wimpy Kid : Rodrick Rules ( February 1 , 2008 ) Diary of a Wimpy Kid : The Last Straw ( January 13 , 2009 ) Diary of a Wimpy Kid : Dog Days (
October 12 , 2009 ) Diary of a Wimpy Kid : The Ugly Truth ( November 9 , 2010 ) Diary of a Wimpy Kid : Cabin Fever ( November 15 , 2011 ) Diary of a Wimpy Kid :
The Third Wheel ( November 13 , 2012 ) Diary of a Wimpy Kid : Hard Luck ( November 5 , 2013 ) Diary of a Wimpy Kid : The Long Haul ( November 4 , 2014 ) Diary
of a Wimpy Kid : Old School ( November 3 , 2015 ) Diary of a Wimpy Kid : Double Down ( November 1 , 2016 ) Diary of a Wimpy Kid : The Getaway ( November 7 ,
</item 8>

<item 9>
Diary of a Wimpy Kid (book series)
ISBN 0 - 8109 - 9313 - 9 . ^ Jump up to : `` '' Diary of a Wimpy Kid ( 2010 ) '' on `` . Comingsoon.net . March 19 , 2010 . Retrieved May 7 , 2011 . ^ Jump up
to : `` Diary of a Wimpy Kid '' . FunBrain . Family Education Network . Retrieved March 5 , 2011 . Jump up ^ `` WIMPY 2 LAST '' . 3 November 2015 . Retrieved 6
March 2016 . Jump up ^ The 2009 TIME 100 : Jeff Kinney Time magazine 's 2009 list of The World 's Most Influential People Jump up ^ Kinney , Jeff . `` The Wimpy
Kid Movie Diary : The Next Chapter '' . Retrieved March 28 , 2017 . Jump up ^ Nickelodeon Kids ' Choice Awards , Nick.com Jump up ^ Dixler , Elsa . `` The New
York Times Best Sellers : Children 's Books '' . The New York Times .
</item 9>

<item 10>
Diary of a Wimpy Kid (book series)
USA Today , which ranks best sellers based on sales alone regardless of genre or intended audience , has listed Diary of a Wimpy Kid in its top 150 , peaking at
number 8 , for 141 weeks as of April 4 , 2010 . As of the same date , Rodrick Rules has been on the list for 117 weeks ( peaking at number 4 ) , The Last Straw
has been on the list for 65 weeks ( peaking at number 1 ) , Dog Days has been on the list for all 25 weeks of its publication ( peaking at number 1 ) , and The
Wimpy Kid Movie Diary has been listed for all three weeks of its publication , peaking at number 2 . The series has sold over 194 million books in more than 30
countries . The series has also generated over $500 million in revenue . Films ( edit ) Main article : Diary of a Wimpy Kid ( film series ) This section needs
to be updated . Please update this article to reflect recent events or newly available information .
</item 10>
=== END ITEM id=77ae4344f2c3bbe9 ===

=== ITEM id=cbb9bcfe552261a0 ===
Gold sentences and facts:
Sentence 0: Originally an ordinary man by the name of Nick Chopper ( the name first appearing in The Marvelous Land of Oz ) , the Tin Woodman used to make his living chopping down trees in the forests of Oz , as his father had before him . The Wicked Witch of the East enchanted his axe to prevent him from marrying his sweetheart , after being bribed by the lazy old woman who kept the Munchkin maiden as a servant , and did not wish to lose her . ( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick 's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no longer able to love the lady he had fallen for .
  Fact 0.0: Nick Chopper was originally an ordinary man.
  Fact 0.1: The name Nick Chopper first appears in The Marvelous Land of Oz.
  Fact 0.2: Nick Chopper used to make his living chopping down trees in the forests of Oz.
  Fact 0.3: Nick Chopper's father had also chopped down trees in the forests of Oz.
  Fact 0.4: The Wicked Witch of the East enchanted Nick Chopper's axe to prevent him from marrying his sweetheart.
  Fact 0.5: The Wicked Witch of the East was bribed to do this by the lazy old woman who kept the Munchkin maiden as a servant.
  Fact 0.6: The lazy old woman did not wish to lose the Munchkin maiden servant.
  Fact 0.7: In a later book of the series, The Tin Woodman of Oz, the woman is said to be the Witch's servant.
  Fact 0.8: In The Tin Woodman of Oz, the Witch herself decides to enchant Nick's axe.
  Fact 0.9: The enchanted axe chopped off Nick Chopper's limbs one by one.
  Fact 0.10: Each time Nick lost a limb, Ku-Klip the tinsmith replaced it with a prosthetic limb made of tin.
  Fact 0.11: Finally, nothing was left of Nick Chopper but tin.
  Fact 0.12: Ku-Klip neglected to replace Nick Chopper's heart.
  Fact 0.13: Once Nick Chopper was made entirely of tin, he was no longer able to love the lady he had fallen for.
Sentence 1: In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest , as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West . The Tin Woodman 's axe proves useful in this journey , both for chopping wood to create a bridge or raft as needed , and for chopping the heads off animals that threaten the party . When the Winged monkeys are sent by the Witch of the West against the group , they throw the Tin Woodman from a great height , damaging him badly . However Winkie Tinsmiths are able to repair him after the death of the Witch .
  Fact 1.0: In The Wonderful Wizard of Oz, Dorothy Gale befriends the Tin Woodman.
  Fact 1.1: Dorothy and her friends find the Tin Woodman rusted in the forest.
  Fact 1.2: The Tin Woodman was caught in rain.
  Fact 1.3: Dorothy uses the Tin Woodman's oil can to release him.
  Fact 1.4: The Tin Woodman follows Dorothy to the Emerald City to get a heart from the Wizard.
  Fact 1.5: The Scarecrow and the Cowardly Lion join Dorothy and the Tin Woodman on their adventure.
  Fact 1.6: The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West.
  Fact 1.7: The Tin Woodman's axe proves useful on this journey for chopping wood to create a bridge or raft as needed.
  Fact 1.8: The Tin Woodman's axe is useful for chopping the heads off animals that threaten the party.
  Fact 1.9: The Winged Monkeys are sent by the Witch of the West against the group.
  Fact 1.10: The Winged Monkeys throw the Tin Woodman from a great height.
  Fact 1.11: The fall damages the Tin Woodman badly.
  Fact 1.12: Winkie Tinsmiths are able to repair the Tin Woodman after the death of the Witch.
Sentence 2: His desire for a heart notably contrasts with the Scarecrow 's desire for brains , reflecting a common debate between the relative importance of the mind and the emotions . This occasions philosophical debate between the two friends as to why their own choices are superior ; neither convinces the other , and Dorothy , listening , is unable to decide which one is right . Symbolically , because they remain with Dorothy throughout her quest , she is provided with both and need not select . The Tin Woodman states unequivocally that he has neither heart nor brain , but cares nothing for the loss of his brain . Towards the end of the novel , though , Glinda praises his brain as not quite that of the Scarecrow 's .
  Fact 2.0: The Tin Woodman's desire for a heart notably contrasts with the Scarecrow's desire for brains.
  Fact 2.1: The contrast reflects a common debate between the relative importance of the mind and the emotions.
  Fact 2.2: The two friends have a philosophical debate about why their own choices are superior.
  Fact 2.3: Neither friend convinces the other in that debate.
  Fact 2.4: Dorothy, listening, is unable to decide which one is right.
  Fact 2.5: Symbolically, because the Tin Woodman and the Scarecrow remain with Dorothy throughout her quest, she is provided with both a heart and a brain and need not select.
  Fact 2.6: The Tin Woodman states unequivocally that he has neither heart nor brain.
  Fact 2.7: The Tin Woodman cares nothing for the loss of his brain.
  Fact 2.8: Towards the end of the novel, Glinda praises the Tin Woodman's brain.
  Fact 2.9: Glinda says the Tin Woodman's brain is not quite that of the Scarecrow.

Retrieved items:
<item 1>
Tin Woodman
Haley also portrayed the Tin Man 's Kansas counterpart , Hickory ( one of Aunt Em and Uncle Henry 's farmworkers ) . Screenwriter Noel Langley created this
character for the film . Hickory helps Zeke ( Lion 's alter ego ) lower a bed into its place on a wagon at the farm . Unlike Zeke , Hickory and Hunk ( Scarecrow
's alter ego ) lose their hats with Uncle Henry as they struggle to open the cellar when the tornado approaches the farm . He reunites with Dorothy when she
awakes from being unconscious . Hickory is seen with Aunt Em , Uncle Henry , Zeke , and Hunk as well as Professor Marvel ( The Wizard 's alter ego ) . In the
original Broadway version of The Wiz , Tiger Haynes played the Tinman , a human woodcutter who became tin after the Wicked Witch of the East cursed his axe to
chop him up ( as in the book ) . Nipsey Russell played the Tinman in the film adaptation of The Wiz .
</item 1>

<item 2>
Tin Woodman
Toward the end , he hugged the Wizard ( Joel Grey ) saying to him `` Thank you from the bottom of my heart ! '' . In 1996 , they made a cartoon animated The Oz
Kids . He has a son named Tin Boy . Like his father he can cut wood with his ax . And he has to be careful with water and snow or he 'll be rust . His voice
played by Andy Milder . Hiroki Tsujiai from Ultra Maniac dresses as the Tin Man in the costume party . James Kall portrayed the Tin Man in the 2001 ABC
miniseries Life with Judy Garland : Me and My Shadows . The Muppet Gonzo plays a similar role , the Tin Thing , in 2005 's television film The Muppets ' Wizard
of Oz . In this version , he is the Wicked Witch 's research assistant , transformed into a robot to prevent him wanting a day off to marry Camilla . Gonzo 's
other role is himself . He appears at the end of this film in the Muppets ' show .
</item 2>

<item 3>
Tin Woodman
The movie was made available for free viewing online and free downloading of its soundtrack . In the 2013 prequel , Oz the Great and Powerful , although the Tin
Woodman does not physically appear , the film introduces his creator , the Master Tinker , portrayed by Bill Cobbs . Tin Man appears in the animated film
Legends of Oz : Dorothy 's Return ( which is based on Dorothy of Oz ) , voiced by Kelsey Grammer . The Woodsman , a 2012 stage play written by James Ortiz with
music by Edward W. Hardy , tells the backstory of the Tin Man , using puppetry , movement , and music . The play has received multiple Off Broadway productions
, and won a 2016 Obie Award for Ortiz 's puppet design . In the TV series Emerald City , the Tin Woodman equivalent is Jack ( Gerran Howell ) , a friend of the
enigmatic Tip , who helps ' him ' escape from his imprisonment by a witch . When it is revealed that Tip is actually a girl whose true identity was suppressed
by a potion , her strained emotional state causes her to push Jack over a balcony after he kisses her .
</item 3>

<item 4>
Tin Woodman
He is also featured in an end credits song entitled `` Rusted . '' Mickey Mouse Clubhouse did a parody special entitled `` The Wizard of Dizz , '' in which the
role of the Tin Man is played by Mickey Mouse . The parody The Muppets ' Wizard of Oz featured Gonzo the Great in the role of the Tin Man . In the game Code
Name : S.T.E.A.M. , Tin Man , along with other Oz characters , is a playable character . His design is more based on Japanese culture and he have the ability to
give steam to allies . The Tin Woodman appears as a major antagonist in Danielle Paige 's Dorothy Must Die novel series . In this series , Dorothy and her
friends have been corrupted by Dorothy 's use of magic and the ' magic ' in the gifts they received from the Wizard , with the Tin Woodman now driven by a
twisted ' love ' for Dorothy that is nevertheless unrequited . He is described as possessing a more twisted appearance , with his legs compared to horses ' legs
and his hands possessing knives for fingers .
</item 4>

<item 5>
Tin Woodman
( In a later book of the series , The Tin Woodman of Oz , the woman is said to be the Witch 's servant , and it is the Witch herself who decides to enchant Nick
's axe . ) The enchanted axe chopped off his limbs , one by one . Each time he lost a limb , Ku - Klip the tinsmith replaced it with a prosthetic limb made of
tin . Finally , nothing was left of him but tin . However , Ku - Klip neglected to replace his heart . Once Nick Chopper was made entirely of tin , he was no
longer able to love the lady he had fallen for . In The Wonderful Wizard of Oz , Dorothy Gale befriends the Tin Woodman after they find him rusted in the forest
, as he was caught in rain , and use his oil can to release him . He follows her to the Emerald City to get a heart from The Wizard . They are joined on their
adventure by the Scarecrow and the Cowardly Lion . The Wizard sends Dorothy and her friends to the Winkie Country to kill the Wicked Witch of the West .
</item 5>

<item 6>
Tin Woodman
) The Magical Mimics in Oz ( 1946 ) The Shaggy Man of Oz ( 1949 ) The Hidden Valley of Oz ( 1951 ) Merry Go Round in Oz ( 1963 ) Yankee in Oz ( 1972 ) The
Enchanted Island of Oz ( 1976 ) The Forbidden Fountain of Oz ( 1980 ) The Ozmapolitan of Oz ( 1986 ) The Wicked Witch of Oz ( 1993 ) The Giant Garden of Oz (
1993 ) The Runaway in Oz ( 1995 ) The Rundelstone of Oz ( 2000 ) The Emerald Wand of Oz ( 2005 ) Trouble Under Oz ( 2006 ) Alternate Dorothy of Oz ( 1989 )
Locations and elements Land of Oz Emerald City Munchkin Country Gillikin Country Winkie Country Quadling Country Yellow brick road Oogaboo Deadly Desert
Nonestica Forest of Burzee Land of Ev Merryland Mo Ix Nome Kingdom Noland Ruby slippers Silver Shoes The Oz Film Manufacturing Company Characters Dorothy Gale
Toto Princess Ozma The Wizard of Oz Scarecrow Tin Woodman Cowardly Lion Jack Pumpkinhead Tik - Tok Sawhorse Trot Hungry Tiger
</item 6>

<item 7>
Tin Woodman
The movie shows him moving through a colorful Oz city with his brand - new mechanical heart , before meeting a reddish female Tin Woman ( or robot ? ) to whom
he gives the heart . She proceeds to cruelly play with the heart . An internet - collaboration , CG animated feature based on Baum 's book The Tin Woodman of Oz
was produced by A : M Films , and completed in 2009 . In 2010 , Whitestone Motion Pictures produced a 23 - minute live - action short film , Heartless : The
Story of the Tin Man . The film is based on the book The Tin Woodman of Oz , where Woot the Wanderer visits the Tin Man and asks how he came to be made of tin .
In the film , the Tin Man appears to be more steam - or coal - powered . His chest cavity is covered by a door which reads `` Pedudoe Tin Co . '' but this is a
reference to the workings of the film company , and not a reference to any Oz book or material . The movie was made available for free viewing online and free
downloading of its soundtrack .
</item 7>

<item 8>
Tin Woodman
They built a story around them , added Dorothy , and added a series of lessons to the effect that everyone possesses the resources they need ( such as brains ,
a heart and courage ) if only they had self - confidence . The Wonderful Wizard of Oz was a children 's book , of course , but as Baum warned in the preface ,
it was a `` modernized '' fairy tale as well . The Tin Man -- the human turned into a machine -- was a common feature in political cartoons and in
advertisements in the 1890s . Indeed , he had been part of European folk art for 300 years . In political interpretations of The Wonderful Wizard of Oz , the
Tin Woodman is supposedly described as a worker , dehumanized by industrialization . The Tin Woodman little by little lost his natural body and had it replaced
by metal ; so he has lost his heart and can not move without the help of farmers ( represented by the Scarecrow ) ; in reality he has a strong sense of
cooperation and love , which needs only an infusion of self - confidence to be awakened . In the 1890s many argued that to secure a political revolution a
coalition of Farmers and Workers was needed .
</item 8>

<item 9>
Tin Woodman
The Wizard turns out to be a `` humbug '' and can only provide a placebo heart made of silk and filled with sawdust . However , this is enough to please the Tin
Woodman , who , with or without a heart , was all along the most tender and emotional of Dorothy 's companions ( just as the Scarecrow was the wisest and the
Cowardly Lion the bravest ) . When he accidentally crushes an insect , he is grief - stricken and , ironically , claims that he must be careful about such
things , while those with hearts do not need such care . This tenderness remains with him throughout the series , as in The Patchwork Girl of Oz , where he
refuses to let a butterfly be maimed for the casting of a spell . When Dorothy returns home to her farm in Kansas , the Tin Woodman returns to the Winkie
Country to rule as emperor . Later , he has his subjects construct a palace made entirely of tin -- from the architecture all the way down to the flowers in the
garden . Baum emphasized that the Tin Woodman remains alive , in contrast to the windup mechanical man Tik - Tok Dorothy meets in a later book .
</item 9>

<item 10>
Tin Woodman
2002 ) Once Upon a Time ( since 2011 ) The Wiz Live ! ( 2015 ) Lost In Oz ( 2015 ) Emerald City ( 2017 ) Dorothy and the Wizard of Oz ( 2017 ) Comics Queer
Visitors from the Marvelous Land of Oz ( 1904 -- 05 ) Adventures in Oz ( 1986 -- 92 ) The Enchanted Apples of Oz ( 1986 ) Oz Squad ( 1991 ) Lost Girls ( 1991 --
92 ) Oz ( 1994 ) Dorothy of Oz ( 2006 ) The Wonderful Wizard of Oz ( 2009 ) Video games The Wizard of Oz ( 1993 ) The Wizard of Oz : Beyond the Yellow Brick
Road ( 2008 ) Emerald City Confidential ( 2009 ) Parodies The Wozard of Iz Os Trapalhões eo Mágico de Oróz The Wonderful Wizard of Ha 's Wizard of Odd
Reimagining The Wizard of Mars ( 1965 ) Hunter ( 1973 ) Zardoz ( 1974 ) Oz ( 1976 ) The Number of the Beast ( 1980 ) A Barnstormer in Oz ( 1982 ) The Wizard of
A.I.D.S.
</item 10>
=== END ITEM id=cbb9bcfe552261a0 ===

=== ITEM id=84d1b0873c7f86cf ===
Gold sentences and facts:
Sentence 0: Douglas Irving Pederson ( born January 31 , 1968 ) is an American football coach who is currently the head coach of the Philadelphia Eagles of the National Football League ( NFL ) . He served as the offensive coordinator of the Kansas City Chiefs from 2013 -- 2015 . He spent most of his playing career as a member of the Green Bay Packers , serving as a backup quarterback to Brett Favre and holder on placekicks , and winning Super Bowl XXXI with the team over the New England Patriots . He was also a backup to Dan Marino as a member of the Miami Dolphins , and a starting quarterback for the Eagles and Cleveland Browns .
  Fact 0.0: Douglas Irving Pederson was born on January 31, 1968.
  Fact 0.1: Douglas Irving Pederson is an American football coach.
  Fact 0.2: Douglas Irving Pederson is currently the head coach of the Philadelphia Eagles.
  Fact 0.3: The Philadelphia Eagles are a team of the National Football League (NFL).
  Fact 0.4: Douglas Irving Pederson served as the offensive coordinator of the Kansas City Chiefs from 2013 to 2015.
  Fact 0.5: Douglas Irving Pederson spent most of his playing career as a member of the Green Bay Packers.
  Fact 0.6: Douglas Irving Pederson served as a backup quarterback to Brett Favre with the Green Bay Packers.
  Fact 0.7: Douglas Irving Pederson served as holder on placekicks with the Green Bay Packers.
  Fact 0.8: Douglas Irving Pederson won Super Bowl XXXI with the Green Bay Packers over the New England Patriots.
  Fact 0.9: Douglas Irving Pederson was also a backup to Dan Marino as a member of the Miami Dolphins.
  Fact 0.10: Douglas Irving Pederson was a starting quarterback for the Philadelphia Eagles.
  Fact 0.11: Douglas Irving Pederson was a starting quarterback for the Cleveland Browns.

Retrieved items:
<item 1>
Doug Pederson
Retrieved 2017 - 08 - 06 . Jump up ^ `` Eagles head coach Doug Pederson gives glory to God , says his faith keeps him grounded '' . Jump up ^ McLane , Jeff (
September 11 , 2016 ) . `` Pederson goes on without his first mentor -- his late father '' . The Philadelphia Inquirer . Retrieved 2017 - 08 - 06 . Jump up ^
Cuellar , Dann . `` Doug Pedersons ' neighbors ready to welcome him back home '' , WPVI - TV , January 19 , 2016 . Accessed November 29 , 2017 . `` New
Philadelphia Eagles head coach Doug Pederson and his family are excited about moving back to the area ... Before moving to Kansas City in 2011 , the Pederson
family lived on a cul - de-sac in the unit block of Hamilton Court in Moorestown , New Jersey . '' External links ( edit ) Philadelphia Eagles profile Current
head coaches of the National Football League American Football Conference AFC East AFC North AFC South AFC West Sean McDermott ( Buffalo Bills ) Adam Gase (
Miami Dolphins ) Bill Belichick ( New England Patriots ) Todd Bowles
</item 1>

<item 2>
Doug Pederson
NFL head coach ( edit ) On January 18 , 2016 , Pederson was hired as head coach of the Eagles replacing Chip Kelly . Despite having Sam Bradford on the roster
as the starting quarterback , the Eagles drafted Carson Wentz with the second overall pick in 2016 . Right before the 2016 season began , Bradford was traded to
the Minnesota Vikings and Wentz was named the starting quarterback as a rookie . Pederson and Wentz won their first three NFL games together , but finished the
season 7 -- 9 , missing the playoffs . Pederson 's second season was much more successful as he led the Eagles to a 13 - 3 record , winning them the NFC East
division championship and allotting them home - field advantage throughout the playoffs . Wentz , who was having a career year and was considered a front runner
for league MVP , tore his ACL in Week 14 , leaving backup Nick Foles with the starting job for the remainder of the year .
</item 2>

<item 3>
Doug Pederson
Calvary was going into its second year as a program when Pederson signed on in March 2005 . Pederson was the head coach at Calvary for four years , and held a
33 -- 7 record in the regular season and an 8 -- 3 record in the post-season . The Cavaliers were in the state playoffs all four years with Pederson as head
coach . In his first season in 2005 , the Cavaliers went 5 -- 6 and lost in the first round of the state playoffs . In 2007 , he led the Cavaliers to the
semi-finals and to their first district title . NFL assistant coaching positions ( edit ) On January 29 , 2009 , Pederson was hired as the offensive quality
control coach for the Philadelphia Eagles , reuniting him with his former head coach , Andy Reid . He was promoted to quarterbacks coach on February 8 , 2011 ,
replacing James Urban , who was promoted to assistant offensive coordinator . On January 11 , 2013 , Pederson followed Andy Reid to the Kansas City Chiefs to
serve as offensive coordinator .
</item 3>

<item 4>
Doug Pederson
625 0 1.000 Coaching tree ( edit ) NFL head coaches under whom Pederson has served : Andy Reid , Philadelphia Eagles ( 2009 -- 2012 ) Kansas City Chiefs ( 2013
- 2015 ) Assistant coaches under Pederson who have become NFL head coaches : Frank Reich , Indianapolis Colts ( 2018 -- present ) Personal life ( edit )
Pederson was born to Teri ( née Boykin ) and Gordon `` Gordy '' Pederson ( 1939 -- 2016 ) on January 31 , 1968 , in Bellingham , Washington . A devout Christian
, Pederson and his wife Jeannie have three sons . Pederson has been a resident of Moorestown , New Jersey . References ( edit ) Jump up ^ Cahill , Dan ( January
25 , 2016 ) . `` Ron Rivera could become one of four to win Super Bowl as player and coach '' . chicago.suntimes.com . Retrieved February 19 , 2018 . Jump up ^
McCallister , Laura ( January 11 , 2013 ) . `` Andy Reid fills out coaching staff , including coordinators '' .
</item 4>

<item 5>
Doug Pederson
-- 2008 ) Head coach Philadelphia Eagles ( 2009 -- 2010 ) Offensive quality control coordinator Philadelphia Eagles ( 2011 -- 2012 ) Quarterbacks coach Kansas
City Chiefs ( 2013 -- 2015 ) Offensive coordinator Philadelphia Eagles ( 2016 -- present ) Head coach Career highlights and awards As player Super Bowl champion
( XXXI ) As coach Super Bowl champion ( LII ) Career NFL statistics Pass attempts : 522 Pass completions : 286 Percentage : 54.7 TD -- INT : 12 -- 19 Passing
yards : 2,762 Passer rating : 62.3 Player stats at NFL.com Head coaching record Regular season : 20 -- 12 (. 625 ) Postseason : 3 -- 0 ( 1.000 ) Career : 23 --
12 (. 657 ) Coaching stats at PFR Douglas Irving Pederson ( born January 31 , 1968 ) is an American football coach who is currently the head coach of the
Philadelphia Eagles of the National Football League ( NFL ) . He served as the offensive coordinator of the Kansas City Chiefs from 2013 -- 2015 .
</item 5>

<item 6>
Doug Pederson
) Andy Reid ( 1999 -- 2012 ) Chip Kelly ( 2013 -- 2015 ) Pat Shurmur # ( 2015 ) Doug Pederson ( 2016 -- ) Pound ( # ) sign denotes interim head coach .
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Doug_Pederson&oldid=832875504 '' Categories : 1968 births Living people American football
quarterbacks Cleveland Browns players Green Bay Packers players High school football coaches in the United States Kansas City Chiefs coaches Louisiana -- Monroe
Warhawks football players Miami Dolphins players New York / New Jersey Knights players People from Moorestown , New Jersey Philadelphia Eagles head coaches
Philadelphia Eagles players Players of American football from Washington ( state ) Rhein Fire players Sportspeople from Bellingham , Washington Super Bowl
champions Super Bowl - winning head coaches Hidden categories : Pages using citations with accessdate and no URL Pages containing links to subscription - only
content Philadelphia Eagles currentteam parameter articles NFL player with coaching information Infobox NFL biography articles missing alt text Talk Contents
About Wikipedia Italiano Português Suomi Edit links This page was last edited on 28 March 2018 , at 13 : 22 .
</item 6>

<item 7>
Doug Pederson
Doug Pederson - wikipedia Doug Pederson Jump to : navigation , search Not to be confused with Doug Peterson . Doug Pederson Pederson in 2016 Philadelphia Eagles
Position : Head coach Born : ( 1968 - 01 - 31 ) January 31 , 1968 ( age 50 ) Bellingham , Washington Height : 6 ft 3 in ( 1.91 m ) Weight : 221 lb ( 100 kg )
Career information High school : Ferndale ( WA ) College : Northeast Louisiana Undrafted : 1991 Career history As player : Miami Dolphins ( 1991 ) * New York /
New Jersey Knights ( 1992 ) Miami Dolphins ( 1992 -- 1994 ) Carolina Panthers ( 1995 ) * Rhein Fire ( 1995 ) Miami Dolphins ( 1995 ) Green Bay Packers ( 1995 --
1998 ) Philadelphia Eagles ( 1999 ) Cleveland Browns ( 2000 ) Green Bay Packers ( 2001 -- 2004 ) * Offseason and / or practice squad member only As coach :
Calvary Baptist Academy ( 2005 -- 2008 ) Head coach Philadelphia Eagles ( 2009 -- 2010 ) Offensive quality
</item 7>

<item 8>
Doug Pederson
In his final start as an Eagle , Pederson was benched at halftime of a week 9 game against the Carolina Panthers after going 3 - of - 9 for 28 yards and being
down 23 -- 0 . He did not see game action at quarterback again until a week 14 game against the Cowboys in which McNabb suffered an injury in the fourth quarter
. Pederson went 8 - for - 12 for 108 yards and a touchdown in the loss , and Koy Detmer received the start ahead of him in week 15 with McNabb still injured .
After spending the next season 's training camp with the team , the Eagles released Pederson on August 28 , 2000 . Cleveland Browns ( edit ) Pederson considered
retirement after being released by the Eagles , but instead signed a two - year contract with the Cleveland Browns on September 2 , 2000 . The Browns ' backup ,
Ty Detmer , suffered a season - ending injury , and the Browns needed a backup quarterback to starter Tim Couch . This was the second time in Pederson 's career
that he was signed to replace an injured Ty Detmer .
</item 8>

<item 9>
Doug Pederson
He spent most of his playing career as a member of the Green Bay Packers , serving as a backup quarterback to Brett Favre and holder on placekicks , and winning
Super Bowl XXXI with the team over the New England Patriots . He was also a backup to Dan Marino as a member of the Miami Dolphins , and a starting quarterback
for the Eagles and Cleveland Browns . In his second season as the Eagles ' head coach , Pederson won Super Bowl LII ( also against the Patriots ) , marking the
first Super Bowl title in franchise history . He also became just the fourth person , after Mike Ditka , Tom Flores , and Tony Dungy to win a Super Bowl as both
a player and head coach . Contents ( hide ) 1 Playing career 1.1 Early years 1.2 Professional 1.2. 1 Miami Dolphins 1.2. 2 First stint with Packers 1.2. 3
Philadelphia Eagles 1.2. 4 Cleveland Browns 1.2.
</item 9>

<item 10>
Doug Pederson
3 Philadelphia Eagles 1.2. 4 Cleveland Browns 1.2. 5 Green Bay Packers ( second stint ) 2 Coaching career 2.1 High school 2.2 NFL assistant coaching positions
2.3 NFL head coach 2.4 Head coaching record 2.5 Coaching tree 3 Personal life 4 References 5 External links Playing career ( edit ) Early years ( edit )
Pederson was born in Bellingham , Washington , in 1968 . He attended Ferndale High School in nearby Ferndale , Washington , and was an All - State selection in
football , basketball , and baseball . After high school he graduated from Northeast Louisiana University , where he was quarterback from 1987 through 1990 . He
still holds multiple passing records at the school . Professional ( edit ) Miami Dolphins ( edit ) Pederson originally signed as a rookie free agent by the
Miami Dolphins on May 1 , 1991 , out of Northeast Louisiana University ( now University of Louisiana at Monroe ) in Monroe , Louisiana . He was waived on August
17 , 1991 , before the start of the regular season .
</item 10>
=== END ITEM id=84d1b0873c7f86cf ===

=== ITEM id=03fbd5f22f7334c1 ===
Gold sentences and facts:
Sentence 0: The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans .
  Fact 0.0: Trade winds are the prevailing pattern of easterly surface winds found in the tropics.
  Fact 0.1: Trade winds are found within the lower portion of the Earth's atmosphere.
  Fact 0.2: Trade winds are found in the lower section of the troposphere near the Earth's equator.
  Fact 0.3: Trade winds blow predominantly from the northeast in the Northern Hemisphere.
  Fact 0.4: Trade winds blow predominantly from the southeast in the Southern Hemisphere.
  Fact 0.5: Trade winds strengthen during the winter.
  Fact 0.6: Trade winds strengthen when the Arctic oscillation is in its warm phase.
  Fact 0.7: Trade winds have been used by captains of sailing ships to cross the world's oceans for centuries.
  Fact 0.8: Trade winds enabled European empire expansion into the Americas.
  Fact 0.9: Trade winds enabled trade routes to become established across the Atlantic and Pacific oceans.

Retrieved items:
<item 1>
Trade winds
Trade winds - wikipedia Trade winds Jump to : navigation , search This article is about the weather phenomenon . For other uses , see Tradewind . The westerlies
( blue arrows ) and trade winds ( yellow and brown arrows ) The trade winds are the prevailing pattern of easterly surface winds found in the tropics , within
the lower portion of the Earth 's atmosphere , in the lower section of the troposphere near the Earth 's equator . The trade winds blow predominantly from the
northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere , strengthening during the winter and when the Arctic oscillation is in
its warm phase . Trade winds have been used by captains of sailing ships to cross the world 's oceans for centuries , and enabled European empire expansion into
the Americas and trade routes to become established across the Atlantic and Pacific oceans . In meteorology , the trade winds act as the steering flow for
tropical storms that form over the Atlantic , Pacific , and southern Indian Oceans and make landfall in North America , Southeast Asia , and Madagascar and
eastern Africa , respectively . Trade winds also transport African dust westward across the Atlantic Ocean into the Caribbean Sea , as well as portions of
southeastern North America .
</item 1>

<item 2>
Trade winds
The subsident ( sinking ) air is relatively dry because as it descends , the temperature increases , but the absolute humidity remains constant , which lowers
the relative humidity of the air mass . This warm , dry air is known as a superior air mass and normally resides above a maritime tropical ( warm and moist )
air mass . An increase of temperature with height is known as a temperature inversion . When it occurs within a trade wind regime , it is known as a trade wind
inversion . The surface air that flows from these subtropical high - pressure belts toward the Equator is deflected toward the west in both hemispheres by the
Coriolis effect . These winds blow predominantly from the northeast in the Northern Hemisphere and from the southeast in the Southern Hemisphere . Because winds
are named for the direction from which the wind is blowing , these winds are called the northeasterly trade winds in the Northern Hemisphere and the
southeasterly trade winds in the Southern Hemisphere . The trade winds of both hemispheres meet at the doldrums . As they blow across tropical regions , air
masses heat up over lower latitudes due to more direct sunlight .
</item 2>

<item 3>
Trade winds
Encyclopedia of world climatology . Springer . p. 128 . ISBN 978 - 1 - 4020 - 3264 - 6 . Retrieved 2009 - 11 - 09 . Jump up ^ Bob Rauber ( 2009 - 05 - 22 ) . ``
Research - The Rain in Cumulus over the Ocean Campaign '' . Retrieved 2009 - 11 - 08 . Jump up ^ James P. Terry ( 2007 ) . Tropical cyclones : climatology and
impacts in the South Pacific . Springer . p. 8 . ISBN 978 - 0 - 387 - 71542 - 1 . Retrieved 2009 - 11 - 08 . Jump up ^ G.E. Pieter & F. Augustinus . `` The
influence of the trade winds on the coastal development of the Guianas at various scale levels : a synthesis '' . Marine Geology. 208 ( 2 - 4 ) : 145 -- 151 .
Bibcode : 2004MGeol. 208 ... 145A .
</item 3>

<item 4>
Trade winds
Those that develop over land ( continental ) are drier and hotter than those that develop over oceans ( maritime ) , and travel northward on the western
periphery of the subtropical ridge . Maritime tropical air masses are sometimes referred to as trade air masses . The one region of the Earth which has an
absence of trade winds is the north Indian ocean . Weather effects ( edit ) Clouds which form above regions within trade wind regimes are typically composed of
cumulus which extend no more than 4 kilometres ( 13,000 ft ) in height , and are capped from being taller by the trade wind inversion . Trade winds originate
more from the direction of the poles ( northeast in the Northern Hemisphere , southeast in the Southern Hemisphere ) during the cold season , and are stronger
in the winter than the summer . As an example , the windy season in the Guianas , which lie at low latitudes in South America , occurs between January and April
. When the phase of the Arctic oscillation ( AO ) is warm , trade winds are stronger within the tropics . The cold phase of the AO leads to weaker trade winds .
</item 4>

<item 5>
Trade winds
They also learned that to reach South Africa , they needed to go far out in the ocean , head for Brazil , and around 30 ° S go east again . Following the
African coast southbound means upwind in the Southern hemisphere . In the Pacific ocean , the full wind circulation , which included both the trade wind
easterlies and higher - latitude Westerlies , was unknown to Europeans until Andres de Urdaneta 's voyage in 1565 . The captain of a sailing ship seeks a course
along which the winds can be expected to blow in the direction of travel . During the Age of Sail , the pattern of prevailing winds made various points of the
globe easy or difficult to access , and therefore had a direct effect on European empire - building and thus on modern political geography . For example ,
Manila galleons could not sail into the wind at all . By the 18th century the importance of the trade winds to England 's merchant fleet for crossing the
Atlantic Ocean had led both the general public and etymologists to identify the name with a later meaning of ' trade ' , `` ( foreign ) commerce '' .
</item 5>

<item 6>
Trade winds
Between 1847 and 1849 , Matthew Fontaine Maury collected enough information to create wind and current charts for the world 's oceans . Cause ( edit ) 3D map
showing Hadley cells in relationship to trade winds on the surface . See also : Air mass , Hadley cell , Humidity , Intertropical Convergence Zone , Monsoon ,
Monsoon trough , Near - equatorial trough , and Prevailing winds As part of the Hadley cell circulation , surface air flows toward the equator while the flow
aloft is towards the poles . A low - pressure area of calm , light variable winds near the equator is known as the doldrums , near - equatorial trough ,
intertropical front , or the Intertropical Convergence Zone . When located within a monsoon region , this zone of low pressure and wind convergence is also
known as the monsoon trough . Around 30 ° in both hemispheres , air begins to descend toward the surface in subtropical high - pressure belts known as
subtropical ridges .
</item 6>

<item 7>
Trade winds
ISBN 978 - 1 - 4437 - 2300 - 8 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Doldrums '' . American Meteorological Society .
Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Intertropical Convergence Zone '' . American Meteorological Society . Retrieved
2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Monsoon Trough '' . American Meteorological Society . Archived from the original on 2009 -
06 - 17 . Retrieved 2009 - 11 - 09 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Superior air '' . American Meteorological Society . Retrieved 2009 -
10 - 28 . Jump up ^ Glossary of Meteorology ( 2009 ) . `` trade winds '' . Glossary of Meteorology .
</item 7>

<item 8>
Trade winds
See also ( edit ) Intertropical Convergence Zone Winds in the Age of Sail Westerly wind burst Volta do mar References ( edit ) Jump up ^ Carol G. Braham ; Enid
Pearsons ; Deborah M. Posner ; Georgia S. Maas & Richard Goodman ( 2001 ) . Random House Webster 's College Dictionary ( second ed . ) . Random House . p. 1385
. ISBN 0 - 375 - 42560 - 8 . Jump up ^ Hermann R. Muelder ( 2007 ) . Years of This Land - A Geographical History of the United States . Read Books . p. 38 .
ISBN 978 - 1 - 4067 - 7740 - 6 . Retrieved 2009 - 11 - 09 . ^ Jump up to : Derek Hayes ( 2001 ) . Historical atlas of the North Pacific Ocean : maps of
discovery and scientific exploration , 1500 - 2000 . Douglas & McIntyre . p. 18 . ISBN 978 - 1 - 55054 - 865 - 5 .
</item 8>

<item 9>
Trade winds
Shallow cumulus clouds are seen within trade wind regimes , and are capped from becoming taller by a trade wind inversion , which is caused by descending air
aloft from within the subtropical ridge . The weaker the trade winds become , the more rainfall can be expected in the neighboring landmasses . Contents ( hide
) 1 History 2 Cause 3 Weather effects 4 See also 5 References History ( edit ) A Spanish galleon See also : Age of Discovery , Volta do Mar , and Age of sail
The term trade winds originally derives from the early fourteenth century late Middle English word ' trade , ' meaning `` path '' or `` track . '' The
Portuguese recognized the importance of the trade winds ( then the Volta do mar , meaning in Portuguese `` turn of the sea '' but also `` return from the sea ''
) in navigation in both the north and south Atlantic ocean as early as the 15th century . From West Africa , the Portuguese had to sail away from continental
Portugal , that is , to west and northwest . They could then turn northeast , to the area around the Azores islands , and finally east to mainland Europe .
</item 9>

<item 10>
Trade winds
`` trade winds '' . Glossary of Meteorology . American Meteorological Society . Archived from the original on 2008 - 12 - 11 . Retrieved 2008 - 09 - 08 . Jump
up ^ Ralph Stockman Tarr and Frank Morton McMurry ( 1909 ). Advanced geography . W.W. Shannon , State Printing , pp. 246 . Retrieved on 2009 - 04 - 15 . Jump up
^ JetStream ( 2008 ) . `` How to read weather maps '' . National Weather Service . Retrieved 2007 - 05 - 16 . Jump up ^ Glossary of Meteorology ( June 2000 ) .
`` Tropical air '' . American Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ Glossary of Meteorology ( June 2000 ) . `` Trade air '' . American
Meteorological Society . Retrieved 2009 - 10 - 28 . Jump up ^ John E. Oliver ( 2005 ) . Encyclopedia of world climatology . Springer .
</item 10>
=== END ITEM id=03fbd5f22f7334c1 ===
