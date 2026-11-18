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

This file has 5 items.

=== ITEM id=a9d49bbe566c817f ===
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
Defamiliarization Taking an everyday object and presenting it in a way that is weirdly unfamiliar so that we see the object in a new way . Coined by the early
20th - century Russian literary critic Viktor Shklovsky in `` Art as Technique . '' In , Swift 's Gulliver 's Travels when Gulliver visits the land of the
giants and sees a giant woman 's skin he sees it is anything but smooth and beautiful when viewed up close . First - person narration A text presented from the
point of view of a character , especially the protagonist , as if the character is telling the story themselves . ( Breaking the fourth wall is an option , but
not a necessity , of this format . ) Mark Twain 's Adventures of Huckleberry Finn uses the title character as the narrator , while Sherlock Holmes is primarily
told from Watson 's perspective . Magical realism Describing events in a real - world setting but with magical trappings , often incorporating local customs and
invented beliefs . Different from urban fantasy in that the magic itself is not the focus of the story . Particularly popular with Latin American authors like
Gabriel García Márquez and Jorge Luis Borges .
</item 1>

<item 2>
List of narrative techniques
If it 's not going to be fired , it should n't be hanging there . '' -- Anton Chekhov Unreliable narrator The narrator of the story is not sincere , or
introduces a bias in their narration and possibly misleads the reader , hiding or minimizing events , characters , or motivations . An example is The Murder of
Roger Ackroyd . The novel includes an unexpected plot twist at the end of the novel . In the last chapter , Sheppard describes how he was an unreliable narrator
. Perspective ( edit ) Name Definition Example Audience surrogate A character who expresses the questions and confusion of the audience , with whom the audience
can identify . Frequently used in detective fiction and science fiction , where the character asks a central character how he or she accomplished certain deeds
, for the purpose of inciting that character to explain ( for the curious audience ) his or her methods , or a character asking a relatively educated person to
explain what amounts to the backstory . Dr Watson in the Sherlock Holmes stories . Scott Evil , played by Seth Green , son of Dr. Evil on the Austin Powers
movies . Author surrogate Characters which are based on authors , usually to support their personal views .
</item 2>

<item 3>
Atticus Finch
This kind of character development , where motivations and ideals between characters , for reasons of plot , are changed is not unusual in the process of
creative writing . Apart from the more progressive depiction of Atticus , the depiction of the town itself , especially the African - American characters , is
also dramatically altered between the two drafts. Real - life comparisons with Lee 's father , Amasa Coleman Lee , have also been made in the two differing
versions of Atticus in that originally Amasa Lee was in favor of segregation but became more liberal later in life , later changing his views to those of
Integration . Tay Hohoff , Lee 's editor , has also been argued to have played a major part in the character development of the novel and particularly Atticus '
liberal transformation . Jonathan Mahler of The New York Times notes in his article The Invisible Hand Behind Harper Lee 's ' To Kill a Mockingbird ' that Ms.
Hohoff , at the same time as she was guiding Ms. Lee through the Mockingbird re-write , was working on her own biography of the early - 20th - century New York
activist and humanist John Lovejoy Elliot .
</item 3>

<item 4>
The passage outlines narrative techniques: foreshadowing plants clues to later events; frame stories and framing devices connect embedded tales (as Scheherazade
does in *Arabian Nights*); a MacGuffin motivates characters despite limited intrinsic importance; in medias res starts mid-action, while a narrative hook grabs
attention. Ochi ends rakugo with a punchline; plot twists overturn expectations; stories within stories embed narratives; ticking-clock scenarios create
urgency; and Chekhov’s gun
</item 4>

<item 5>
List of narrative techniques
Author surrogate Characters which are based on authors , usually to support their personal views . Sometimes an intentionally or unintentionally idealized
version of them . A variation is the Mary Sue or Gary Stu , which primarily serves as an idealized self - insertion . Socrates in the writings of Plato . Plato
never speaks in his own voice in his dialogues . In the Second Letter , it says , `` no writing of Plato exists or ever will exist , but those now said to be
his are those of a Socrates become beautiful and new '' . Breaking the fourth wall An author or character addresses the audience directly ( also known as direct
address ) . This may acknowledge to the reader or audience that what is being presented is fiction , or may seek to extend the world of the story to provide the
illusion that they are included in it . The characters in Sesame Street often break the fourth wall when they address their viewers as part of the ongoing
storyline , which is possible because of the high level of suspension of belief afforded by its audience -- children . The American political drama show House
of Cards also uses this technique frequently to let the viewers know what the main character Frank Underwood is thinking and planning .
</item 5>

<item 6>
List of narrative techniques
This technique frequently occurs within a poetic line grammatically connected to the end of the previous line by enjambment . E.g. , in `` Know then thyself . ‖
Presume not God to scan . '' Distancing effect Deliberately preventing the audience from identifying with characters in order to let them be coolly scrutinized
. Popularized by 20th century playwright Bertolt Brecht . Dramatic visualization Representing an object or character with abundant descriptive detail , or
mimetically rendering gestures and dialogue to make a scene more visual or imaginatively present to an audience . This technique appears at least as far back as
the Arabian Nights . Euphuism An artificial , highly elaborate way of writing or speaking . Named from Euphues ( 1579 ) the prose romance by John Lyly . `` Is
it not far better to abhor sins by the remembrance of others ' faults , than by repentance of thine own follies ? '' ( Euphues , 1 , lecture by the wise
Neapolitan ) Hyperbole Exaggeration used to evoke strong feelings or create an impression which is not meant to be taken literally .
</item 6>

<item 7>
List of narrative techniques
Her tears were a river flowing down her cheeks . Thematic patterning Distributing recurrent thematic concepts and moralistic motifs among various incidents and
frames of a story . In a skillfully crafted tale , thematic patterning may emphasize the unifying argument or salient idea disparate events and disparate frames
have in common . Each of the chapters of Ulysses by James Joyce . Character ( edit ) Name Type Notes Anthropomorphism Form of personification that applies human
- like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis
Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw .
Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in
the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors
Lear 's mental deterioration .
</item 7>

<item 8>
List of narrative techniques
Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II
of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . '' References ( edit ) Jump up ^ Orehovec , Barbara ( 2003 ) .
Revisiting the Reading Workshop : A Complete Guide to Organizing and Managing an Effective Reading Workshop That Builds Independent , Strategic Readers (
illustrated ed . ) . Scholastic Inc. p. 89 . ISBN 0439444047 . Jump up ^ Demchick , Harrison ( 2013 - 09 - 26 ) . `` Techniques and Tension in Breaking Bad '' .
Huffington Post . Retrieved 2017 - 11 - 14 . Jump up ^ `` Literature Glossary - Defamiliarization '' . www.shmoop.com . Retrieved 2017 - 11 - 14 . Jump up ^ ``
Allegory Examples '' . YourDictionary . Retrieved 2017 - 11 - 14 .
</item 8>

<item 9>
List of narrative techniques
His parents died protecting him , and when Voldemort tried to cast a killing curse on Harry , it rebounded and took away most of his strength , and gave Harry
Potter a unique ability and connection with the Dark Lord thus marking him as his equal . Story within a story ( Hypodiegesis ) A story told within another
story . See also frame story . In Stephen King 's The Wind Through the Keyhole , of the Dark Tower series , the protagonist tells a story from his past to his
companions , and in this story he tells another relatively unrelated story . Ticking clock scenario Threat of impending disaster -- often used in thrillers
where salvation and escape are essential elements In the TV show `` 24 '' , the main character , Jack Bauer often finds himself interrogating a terrorist who is
caught in order to disarm a bomb . Chekhov 's gun A dramatic principle that requires every element in a narrative to be irreplaceable , with anything else
removed . `` Remove everything that has no relevance to the story . If you say in the first chapter that there is a rifle hanging on the wall , in the second or
third chapter it absolutely must go off .
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a9d49bbe566c817f ===

=== ITEM id=47686c7e5eaa769f ===
Gold sentences and facts:
Sentence 0: In August 2015 , the series was reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop other series with Marvel instead .
  Fact 0.0: In August 2015, the series was reinvented as Most Wanted.
  Fact 0.1: The series Most Wanted received a pilot order from ABC in August 2015.
  Fact 0.2: The cast was filled out in early 2016.
  Fact 0.3: Production was completed by late March.
  Fact 0.4: The episode was co-produced by ABC Studios and Marvel Television.
  Fact 0.5: In May 2016, ABC announced that it would not pick up the pilot to series.
  Fact 0.6: In May 2016, ABC announced that it would look to develop other series with Marvel instead.

Retrieved items:
<item 1>
Most Wanted (TV pilot)
Most Wanted ( TV pilot ) - wikipedia Most Wanted ( TV pilot ) Jump to : navigation , search Most Wanted Genre Action / Adventure Science fiction Superhero Drama
Created by Jeffrey Bell Paul Zbyszewski Based on Mockingbird by Gerry Conway Barry Smith Lance Hunter by Gary Friedrich Herb Trimpe Starring Adrianne Palicki
Nick Blood Delroy Lindo Fernanda Andrade Oded Fehr Country of origin United States Original language ( s ) English Production Executive producer ( s ) Jeffrey
Bell Paul Zbyszewski Jeph Loeb Jim Chory Production company ( s ) ABC Studios Marvel Television Distributor Disney -- ABC Domestic Television Release Original
network ABC Picture format 720p ( HDTV ) Audio format 5.1 surround sound Chronology Related shows Marvel 's Agents of S.H.I.E.L.D. Marvel Cinematic Universe
television series Marvel 's Most Wanted , or simply Most Wanted , is an unaired American television pilot based on the Marvel Comics characters Bobbi Morse and
Lance Hunter .
</item 1>

<item 2>
Most Wanted (TV pilot)
^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) . `` How Will ' Agents of SHIELD ' Set Up Spinoff ' Marvel 's Most Wanted ' ? '' . The Hollywood Reporter
. Archived from the original on August 20 , 2016 . Retrieved August 20 , 2016 . Jump up ^ Carinacas , Peter ( February 17 , 2016 ) . `` Artisans Booked on '
Marvel 's Most Wanted , ' ' Pearl , ' ' Dirty Dancing ' and ' Stronger ' '' . Variety . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017
. Jump up ^ Carinacas , Peter ( March 15 , 2016 ) . `` Agencies Book Artisans on ' Thor : Ragnarok , ' ' Sugar Queen , ' ' Chagrin Falls ' and ' Bunker Hill '
'' . Variety . Archived from the original on March 20 , 2016 . Retrieved March 20 , 2016 .
</item 2>

<item 3>
Most Wanted (TV pilot)
Archived from the original on February 23 , 2016 . Retrieved February 22 , 2016 . ^ Jump up to : `` Oded Fehr and Fernanda Andrade Join ' Marvel 's Most Wanted
' '' . Marvel.com . March 25 , 2016 . Archived from the original on March 26 , 2016 . Retrieved March 25 , 2016 . ^ Jump up to : Khatchatourian , Maane (
February 14 , 2015 ) . `` ' Agents of SHIELD ' Promotes Adrianne Palicki to Series Regular '' . Variety . Archived from the original on February 15 , 2015 .
Retrieved February 15 , 2015 . ^ Jump up to : Abrams , Natalie ( March 23 , 2016 ) . `` Agents of S.H.I.E.L.D. 's Adrianne Palicki , Nick Blood on that
emotional exit '' . Entertainment Weekly . Archived from the original on March 25 , 2016 . Retrieved March 25 , 2016 .
</item 3>

<item 4>
Most Wanted (TV pilot)
In August 2015 , the spin - off series received new life , reworked as Marvel 's Most Wanted and given a pilot order . Bell and Zbyszewski were again developing
the series , writing the pilot and executive producing . The pair were set to serve as showrunners for the series , if it was picked up . Loeb and Jim Chory
were also attached as executive producers . The series would still focus on Morse and Hunter , with Palicki and Blood both attached , and was described as `` a
new take focusing on the same duo and their continuing adventures . '' In January 2016 , Lee stated that the pilot was `` absolutely '' moving forward ,
praising the script and saying production would begin `` in the next few months . '' In the Agents of S.H.I.E.L.D. episode `` Parting Shot '' , the two
characters were written off that series , having them leave and disavow S.H.I.E.L.D.
</item 4>

<item 5>
Most Wanted (TV pilot)
The series would have been based on storylines occurring at the end of that second season , but ABC passed on the project . In August 2015 , the series was
reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode
was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop
other series with Marvel instead . Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Main 2.2 Guests 3 Production 3.1 Development 3.2 Writing 3.3 Casting
3.4 Filming 3.5 Marvel Cinematic Universe tie - ins 4 No pick - up and future 5 References 6 External links Premise ( edit ) Ex-spies and ex-spouses Bobbi Morse
and Lance Hunter are on the run trying to uncover a conspiracy against them . With no help from S.H.I.E.L.D.
</item 5>

<item 6>
Most Wanted (TV pilot)
Strange Captain America Captain America II : Death Too Soon The Incredible Hulk Returns The Trial of the Incredible Hulk The Death of the Incredible Hulk Power
Pack Generation X Nick Fury : Agent of S.H.I.E.L.D. Man - Thing See also Marvel Television List of Marvel Cinematic Universe television series Marvel Cinematic
Universe portal Television in the United States portal Superhero fiction portal 2010s portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Most_Wanted_(TV_pilot)&oldid=800040909 '' Categories : American action television series American drama television
series American science fiction television series English - language television programs Espionage television series Marvel Cinematic Universe television
programs Serial drama television series Television pilots not picked up as a series Television series by ABC Studios Television spin - offs Unaired television
pilots Hidden categories : Good articles Talk Contents About Wikipedia Suomi 中文 Edit links This page was last edited on 11 September 2017 , at 07 : 06 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy .
</item 6>

<item 7>
Most Wanted (TV pilot)
^ Jump up to : Schwartz , Terri ( March 22 , 2016 ) . `` Agents of SHIELD EP Jeff Bell on Mockingbird and Hunter Departure and Marvel 's Most Wanted '' . IGN .
Archived from the original on March 23 , 2016 . Retrieved March 23 , 2016 . Jump up ^ Schwartz , Terri ( January 9 , 2016 ) . `` Agents of SHIELD Showrunners
Explain Marvel 's Most Wanted Spinoff Plan '' . IGN . Archived from the original on January 10 , 2016 . Retrieved January 10 , 2016 . ^ Jump up to : Lovett ,
Jamie ( October 27 , 2015 ) . `` EXCLUSIVE Agents Of S.H.I.E.L.D. EP Talks Secret Warriors , Civil War , Last Week 's Shocking Death & More '' . ComicBook.com .
Archived from the original on October 27 , 2015 . Retrieved October 28 , 2015 . ^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) .
</item 7>

<item 8>
Most Wanted (TV pilot)
^ Jump up to : `` Marvel 's Most Wanted '' . MckennaGrace.com . May 3 , 2016 . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017 . ^ Jump
up to : Frederick , Brittany . `` Who directed Spider - Man : Homecoming ? '' . Fansided . Archived from the original on July 14 , 2017 . Retrieved July 14 ,
2017 . ^ Jump up to : Walsh , S.M. ( June 6 , 2016 ) . `` ' Stitchers ' Season 3 : Meet the Cast '' . Heavy.com . Archived from the original on July 14 , 2017 .
Retrieved July 14 , 2017 . Jump up ^ Daniels , Hunter ( July 23 , 2011 ) . `` Comic - Con : Marvel TV Announces Development Slate ; Includes Live - Action AKA
Jessica Jones ( Alias ) , The Hulk , Cloak And Dagger , Mockingbird , Animated Hulk And The Agents Of S.M.A.S.H. , More '' .
</item 8>

<item 9>
Most Wanted (TV pilot)
Retrieved March 25 , 2016 . ^ Jump up to : Abrams , Natalie ( July 25 , 2014 ) . `` Agents of S.H.I.E.L.D. Reveals New Characters , Carter Adds Captain
Directors '' . TV Guide . Archived from the original on July 26 , 2014 . Retrieved July 25 , 2014 . ^ Jump up to : Andreeva , Nellie ( January 26 , 2016 ) . ``
' Marvel 's Most Wanted ' ABC Pilot Casts Delroy Lindo As Dominic Fortune '' . Deadline.com . Archived from the original on January 26 , 2016 . Retrieved
January 27 , 2016 . Jump up ^ Huver , Scott ( April 22 , 2016 ) . `` '' Marvel 's Most Wanted '' Co-Star Andrade Promises An `` Unpredictable '' Series `` .
Comic Book Resources . Archived from the original on April 22 , 2016 . Retrieved April 22 , 2016 . ^ Jump up to : `` Marvel 's Most Wanted '' .
</item 9>

<item 10>
Most Wanted (TV pilot)
Retrieved August 9 , 2016 . External links ( edit ) Marvel 's Most Wanted on IMDb Marvel Cinematic Universe Feature films Phase One Iron Man The Incredible Hulk
Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The Winter Soldier
Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2 Spider - Man :
Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television series ABC
Agents of S.H.I.E.L.D. episodes season 1 5 characters Agent Carter season 1 characters Inhumans `` Behold ... The Inhumans '' `` Those Who Would Destroy Us ''
Netflix Daredevil season 1 characters Jessica Jones season 1 characters Luke Cage season 1 characters Iron Fist season 1 characters The Defenders The Punisher
Hulu Runaways Freeform Cloak & Dagger Unaired pilot Most Wanted Other Short films Item 47 Agent Carter All Hail the King Digital series WHIH Newsfront Agents of
S.H.I.E.L.D.
</item 10>
=== END ITEM id=47686c7e5eaa769f ===

=== ITEM id=061bfcfbd3b1b005 ===
Gold sentences and facts:
Sentence 0: In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript 3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder ( built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of components that included charting , advanced UI , and data services ( Flex Data Services ) .
  Fact 0.0: In 2007, Adobe's first version release was Adobe Flash CS3 Professional.
  Fact 0.1: Adobe Flash CS3 Professional was the ninth major version of Flash.
  Fact 0.2: Adobe Flash CS3 Professional introduced the ActionScript 3.0 programming language.
  Fact 0.3: ActionScript 3.0 supported modern programming practices.
  Fact 0.4: ActionScript 3.0 enabled business applications to be developed with Flash.
  Fact 0.5: Adobe Flex Builder was built on Eclipse.
  Fact 0.6: Adobe Flex Builder targeted the enterprise application development market.
  Fact 0.7: Adobe Flex Builder was released in 2007.
  Fact 0.8: Flex Builder included the Flex SDK.
  Fact 0.9: The Flex SDK is a set of components that included charting, advanced UI, and data services.
  Fact 0.10: The data services in the Flex SDK are called Flex Data Services.

Retrieved items:
<item 1>
Adobe Flash
, blend modes ( similar to Adobe Photoshop ) , and advanced features for FLV video . Adobe ( edit ) Macromedia was acquired by Adobe Systems in 2005 , and the
entire Macromedia product line including Flash , Dreamweaver , Director / Shockwave , Fireworks ( which has since been discontinued ) and Authorware is now
handled by Adobe . In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript
3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder (
built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of
components that included charting , advanced UI , and data services ( Flex Data Services ) . In 2008 , Adobe released the tenth version of Flash , Adobe Flash
CS4 . Flash 10 improved animation capabilities within the Flash editor , adding a motion editor panel ( similar to Adobe After Effects ) , inverse kinematics (
bones ) , basic 3D object animation , object - based animation , and other text and graphics features .
</item 1>

<item 2>
Adobe Flash
CreateJS is library that while available separately was also adopted by Adobe as a replacement for Wallaby in CS6 . Unlike Wallaby , which was a standalone
program , the `` Toolkit for CreateJS '' only works as a plug - in inside Flash Professional ; it generates output for the HTML5 canvas , animated with
JavaScript . Around December 2013 , the toolkit was integrated directly into Flash Professional CC . Criticisms ( edit ) Mobile support ( edit ) Websites built
with Adobe Flash will not function on most modern mobile devices running Google Android or iOS ( iPhone , iPad ) . The only alternative is using HTML5 and
responsive web design to build websites that support both desktop and mobile devices . However , Flash is still used to build mobile games using Adobe AIR .
Such games will not work in mobile web browsers , but must be installed via the appropriate app store . Vendor dependence ( edit ) See also : Vendor lock - in
The reliance on Adobe for decoding Flash makes its use on the World Wide Web a concern -- the completeness of its public specifications are debated , and no
complete implementation of Flash is publicly available in source code form with a license that permits reuse .
</item 2>

<item 3>
Adobe Flash
Previously , developers could not use the specification for making SWF - compatible players , but only for making SWF - exporting authoring software . The
specification still omits information on codecs such as Sorenson Spark , however . Animation tools ( edit ) Official tools ( edit ) Main article : Adobe Animate
The Adobe Animate authoring program is primarily used to design graphics and animation and publish the same for websites , web applications , and video games .
The program also offers limited support for audio and video embedding , and ActionScript scripting . Adobe released Adobe LiveMotion , designed to create
interactive animation content and export it to a variety of formats , including SWF . LiveMotion failed to gain any notable user base . In February 2003 ,
Macromedia purchased Presedia , which had developed a Flash authoring tool that automatically converted PowerPoint files into Flash . Macromedia subsequently
released the new product as Breeze , which included many new enhancements .
</item 3>

<item 4>
Adobe Flash
The Apache Flex ActionScript 3.0 compiler generates SWF files from ActionScript 3 files . Flex was the primary ActionScript 3 compiler and was actively
developed by Adobe before it was donated to Apache Software Foundation in 2011 . Adobe Animate -- primarily used to design graphics and animation , but supports
ActionScript scripting and debugging . Adobe Flash Builder -- enterprise application development & debugging , contains the Flex SDK with UI and charting
components . Adobe Scout -- a visual profiler to optimize performance of Flash content . CrossBridge -- a free SDK to cross-compile C++ code to run in Flash
Player . Third - party tools ( edit ) Third - party development tools have been created to assist developers in creating software applications and video games
with Flash . FlashDevelop is a free and open - source Flash ActionScript IDE , which includes a project manager and debugger for building applications on Flash
Player and Adobe AIR . Powerflasher FDT is a commercial ActionScript IDE similar to FlashDevelop . Haxe is an open source , high - level object - oriented
programming language geared towards web - content creation that can compile SWF files from Haxe programs .
</item 4>

<item 5>
Adobe Flash
Adobe Wallaby Server - side Adobe Flash Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC
Scaleform GFx SWiSH Max Swift 3D Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash
controversy Flash animation films television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software
Macromedia Adobe Systems Adobe Creative Suite and Creative Cloud Current products Acrobat After Effects Animate Audition Bridge Dreamweaver Flash Builder
Illustrator InCopy InDesign Lightroom Media Encoder Muse Photoshop Prelude Premiere Pro Spark XD Creative Cloud services Behance Business Catalyst Story Typekit
Former products Contribute Device Central Edge Edge Animate Encore Flash Catalyst Fireworks GoLive ImageReady OnLocation Soundbooth Stock Photos Ultra Adobe
eLearning Suite Current products Captivate Flash Dreamweaver Photoshop Acrobat Presenter Soundbooth Shared applications Bridge Device Central Adobe Systems
Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code Edge
Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker
Photoshop
</item 5>

<item 6>
Adobe Flash Player
( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark Gameswf Tamarin Animation tools Adobe Animate Toon Boom Moho GoAnimate
Stencyl Programming tools FlashDevelop Powerflasher FDT Apache Flex Haxe CrossBridge Adobe Scout FlashFirebug Libraries Starling Framework Away3D Flare3D
Stage3D PureMVC Ming List of Flex frameworks Converters Adobe Flash Media Live Encoder SWFTools swfmill Google Swiffy Adobe Wallaby Server - side Adobe Flash
Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC Scaleform GFx SWiSH Max Swift 3D
Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash controversy Flash animation films
television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software Macromedia Adobe Systems Adobe
Systems Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code
Edge Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse
</item 6>

<item 7>
Adobe Flash
Artists may produce Flash graphics and animations using Adobe Animate . Software developers may produce applications and video games using Adobe Flash Builder ,
FlashDevelop , Flash Catalyst , or any text editor when used with the Apache Flex SDK . End - users can view Flash content via Flash Player ( for web browsers )
, AIR ( for desktop or mobile apps ) or third - party players such as Scaleform ( for video games ) . Adobe Flash Player ( supported on Microsoft Windows ,
macOS and Linux ) enables end - users to view Flash content using web browsers . Adobe Flash Lite enabled viewing Flash content on older smartphones , but has
been discontinued and superseded by Adobe AIR . The ActionScript programming language allows the development of interactive animations , video games , web
applications , desktop applications and mobile applications . Programmers can implement Flash software using an IDE such as Adobe Animate , Adobe Flash Builder
, Adobe Director , FlashDevelop and Powerflasher FDT . Adobe AIR enables full - featured desktop and mobile applications to be developed with Flash and
published for Windows , macOS , Android , iOS , Xbox One , PlayStation 4 , and Nintendo Wii U and Switch .
</item 7>

<item 8>
Adobe Flash
FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker Photoshop Photoshop Lightroom Premiere Pro Reader
Shockwave Player Server LiveCycle Flash Media Server BlazeDS ColdFusion JRun PhoneGap Build Technologies ActionScript Adobe AIR Adobe Flash Adobe Flex Adobe
Shockwave Adobe Font Folio Digital Negative ( DNG ) Authorware FlashPaper Portable Document Format ( PDF ) PostScript Web services Acrobat.com Adobe Solutions
Network Adobe Photoshop Express Adobe Premiere Express Adobe Spark Adobe Stock Acquisitions Aldus Macromedia Scene7 Omniture Behance Fotolia People Bruce Chizen
Charles Geschke Shantanu Narayen John Warnock Del Yocam Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Adobe_Flash&oldid=821675124 ''
Categories : Adobe Flash American inventions Cross-platform software Computing platforms C++ software Mac OS multimedia software MacOS multimedia software
Windows multimedia software Macromedia software Adobe software 1993 software Video game development software Hidden categories : Webarchive template wayback
links Pages using web citations with no URL Pages using citations with accessdate and no URL Use mdy dates from April 2012 Articles containing potentially dated
statements from February 2009 All articles containing potentially dated statements Articles containing potentially
</item 8>

<item 9>
Adobe Flash
Adobe AIR was upgraded to support 64 - bit computers , and to allow developers to add additional functionality to the AIR runtime using AIR Native Extensions (
ANE ) . In 2014 , Adobe AIR reached a milestone with over 100,000 unique applications built , and over 1 billion installations logged across the world ( May
2014 ) . Adobe AIR was voted the Best Mobile Application Development product at the Consumer Electronics Show on two consecutive years ( CES 2014 and CES 2015 )
. In 2016 , Adobe renamed Flash Professional , the primary authoring software for Flash content , to Adobe Animate to reflect its growing use for authoring
HTML5 content in favour of Flash content . Open Screen Project ( edit ) On May 1 , 2008 , Adobe announced the Open Screen Project , with the intent of providing
a consistent application interface across devices such as personal computers , mobile devices , and consumer electronics .
</item 9>

<item 10>
Adobe Flash
Adobe Flash - Wikipedia Adobe Flash Jump to : navigation , search For the application / animation - maker , see Adobe Animate . For other uses , see Adobe Flash
( disambiguation ) . Adobe Flash Developer ( s ) Adobe Systems Macromedia FutureWave Target platform ( s ) Web browsers , iOS , Android , Windows , macOS Editor
software Adobe Animate , Flash Builder , FlashDevelop , Powerflasher FDT , Flash Catalyst Scaleform Player software Flash Player Adobe AIR OpenFL Scaleform
Gnash Lightspark Format ( s ) SWF FLV FLA Programming language ( s ) ActionScript Application ( s ) Animation Browser games Rich Internet applications Desktop
applications Mobile apps Mobile games console games computer games Status Active ( Until 2020 ) License Proprietary Adobe Flash is a multimedia software
platform used for production of animations , rich Internet applications , desktop applications , mobile applications , mobile games and embedded web browser
video players . Flash displays text , vector graphics and raster graphics to provide animations , video games and applications . It allows streaming of audio
and video , and can capture mouse , keyboard , microphone and camera input . Artists may produce Flash graphics and animations using Adobe Animate .
</item 10>
=== END ITEM id=061bfcfbd3b1b005 ===

=== ITEM id=37d81753f25bc627 ===
Gold sentences and facts:
Sentence 0: The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season . Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 .
  Fact 0.0: The College Football Playoff selection committee chose the semifinalists.
  Fact 0.1: The semifinalists were chosen following the conclusion of the 2017 regular season.
  Fact 0.2: Alabama advanced to the national championship.
  Fact 0.3: Georgia advanced to the national championship.
  Fact 0.4: Alabama won its semifinal game hosted by the Sugar Bowl on January 1, 2018.
  Fact 0.5: Georgia won its semifinal game hosted by the Rose Bowl on January 1, 2018.
  Fact 0.6: The championship game was played at Mercedes-Benz Stadium in Atlanta, Georgia.
  Fact 0.7: The championship game was played on January 8, 2018.
Sentence 1: The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National Championship on November 4 , 2015 .
  Fact 1.0: Mercedes-Benz Stadium in Atlanta, Georgia was announced as the host site for the fourth College Football National Championship.
  Fact 1.1: The host site announcement was made on November 4, 2015.

Retrieved items:
<item 1>
2018 College Football Playoff National Championship
Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on
January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 . Contents ( hide ) 1 Background 2 2018
College Football Playoff 3 Teams 3.1 Alabama 3.2 Georgia 4 Game summary 4.1 Scoring summary 4.2 Statistics 5 Broadcasting 6 Dispute 7 See also 8 References 9
External links Background ( edit ) The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National
Championship on November 4 , 2015 .
</item 1>

<item 2>
2018 College Football Playoff National Championship
related to 2018 College Football Playoff National Championship at Wikimedia Commons College Football Playoff Championships Overview College Football Playoff
National Championship Games 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2019 -- 20 2020 -- 21 Semifinal Bowls Rose 2014 ( Jan 2015 ) 2017 ( Jan 2018
) 2020 ( Jan 2021 ) Sugar 2014 ( Jan 2015 ) 2017 ( Jan 2018 ) 2020 ( Jan 2021 ) Orange 2015 ( Dec 2015 ) 2018 ( Dec 2018 ) Cotton 2015 ( Dec 2015 ) 2018 ( Dec
2018 ) Peach 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Fiesta 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Other List of College Football Playoff games New Year 's Six College
Football Playoff National Championship Trophy NCAA Division I FBS champions Mythical national championship Plus - One system Power Five conferences Championship
games for each season are played in
</item 2>

<item 3>
2018 College Football Playoff National Championship
2018 College football Playoff ( edit ) Semifinals Championship January 1 -- Sugar Bowl Mercedes - Benz Superdome , New Orleans 1 Clemson 6 4 Alabama 24 January
8 -- National Championship Mercedes - Benz Stadium , Atlanta 4 Alabama ( OT ) 26 January 1 -- Rose Bowl Rose Bowl , Pasadena 3 Georgia 23 2 Oklahoma 48 3
Georgia ( 2OT ) 54 This bracket : view talk edit The College Football Playoff ( CFP ) selected four teams to advance to the playoff : the Clemson Tigers ,
Oklahoma Sooners , Georgia Bulldogs , and Alabama Crimson Tide . Teams ( edit ) Alabama ( edit ) See also : 2017 Alabama Crimson Tide football team The Alabama
Crimson Tide defeated the Clemson Tigers in the 2018 Sugar Bowl by a score of 24 -- 6 to reach the championship game . The Tide had a 12 -- 1 season , becoming
the SEC West Division co-champions with the Auburn Tigers . The Crimson Tide were coached by Nick Saban .
</item 3>

<item 4>
2018 College Football Playoff National Championship
Jump up ^ Alex Scarbrough ( November 4 , 2015 ) . `` Atlanta , Santa Clara and New Orleans land CFP title games for 2018 -- 20 '' . ESPN.com . Retrieved
November 4 , 2015 . ^ Jump up to : `` Selection Committee Rankings : Final Top 25 Rankings '' ( PDF ) . College Football Playoff . December 3 , 2017 . Retrieved
December 3 , 2017 . Jump up ^ `` Alabama 's Sugar Bowl win against Clemson sets up all - SEC final : Final score , stats , analysis '' . nola.com . January 1 ,
2018 . Retrieved January 2 , 2018 . Jump up ^ Emerson , Seth ( January 1 , 2018 ) . `` Georgia to play for national championship after Rose Bowl win in 2 OT ''
. ajc.com . Retrieved January 2 , 2018 . ^ Jump up to : Hoffman , Benjamin ; Drape , Joe ; Tracy , Marc ( January 8 , 2018 ) . `` National Championship Game :
Georgia vs.
</item 4>

<item 5>
2018 College Football Playoff National Championship
28 ) Camping World ( Dec. 28 ) Alamo ( Dec. 28 ) Holiday ( Dec. 28 ) Belk ( Dec. 29 ) Cotton ( Dec. 29 ) Sun ( Dec. 29 ) Music City ( Dec. 29 ) Arizona ( Dec.
29 ) Fiesta ( Dec. 30 ) Liberty ( Dec. 30 ) Orange ( Dec. 30 ) TaxSlayer ( Dec. 30 ) Citrus ( Jan. 1 ) Outback ( Jan. 1 ) Peach ( Jan. 1 ) College Football
Playoff games : Rose ( semifinal ) ( Jan. 1 ) Sugar ( semifinal ) ( Jan. 1 ) National Championship Game ( Jan. 8 ) All - Star Games East -- West Shrine Game (
Jan. 20 ) NFLPA Collegiate Bowl ( Jan. 20 ) Senior Bowl ( Jan. 27 ) NCAA Football Championship Games Division I FBS 2007 * 2008 * 2009 * 2010 * 2011 * 2012 *
2013 * 2014 * 2015 * 2016 * 2017 * 2018 * 2019 *
</item 5>

<item 6>
2018 College Football Playoff National Championship
Alabama Defensive : # 94 DT Da'Ron Payne , Jr . Alabama Favorite Alabama by 3.5 National anthem Zac Brown Band Referee Dan Capron ( Big Ten ) Halftime show
Georgia Redcoat Marching Band Million Dollar Band Attendance 77,430 United States TV coverage Network ESPN / ESPN Radio / ESPN Deportes Announcers Chris Fowler
, Kirk Herbstreit , Maria Taylor and Tom Rinaldi ( ESPN ) Sean McDonough , Todd Blackledge , Holly Rowe and Ian Fitzsimmons ( ESPN Radio ) College Football
Playoff National Championship < 2017 2019 > The 2018 College Football Playoff National Championship was a college football bowl game that determined the
national champion in the NCAA Division I Football Bowl Subdivision for the 2017 season . The Alabama Crimson Tide defeated the Georgia Bulldogs 26 -- 23 in
overtime . Alabama overcame a 13 -- 0 deficit at halftime . Tua Tagovailoa and Da'Ron Payne were respectively named the offensive and defensive players of the
game . The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season .
</item 6>

<item 7>
2018 College Football Playoff National Championship
Not an officially sanctioned NCAA championship Retrieved from ``
https://en.wikipedia.org/w/index.php?title=2018_College_Football_Playoff_National_Championship&oldid=822294017 '' Categories : 2017 -- 18 NCAA football bowl
games 2017 -- 18 College Football Playoff 2018 in sports in Georgia ( U.S. state ) Sports competitions in Atlanta American football in Georgia ( U.S. state )
2010s in Atlanta Alabama Crimson Tide football bowl games Georgia Bulldogs football bowl games January 2018 sports events in the United States Hidden categories
: Use mdy dates from January 2018 Talk Contents About Wikipedia Français Edit links This page was last edited on 25 January 2018 , at 15 : 15 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
2018 College Football Playoff National Championship
1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018
Division III 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007
2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Games played in January are denoted by an asterisk Alabama Crimson Tide bowl games 1926 Rose Bowl 1927 Rose
Bowl 1931 Rose Bowl
</item 8>

<item 9>
2018 College Football Playoff National Championship
2018 College football Playoff national championship - wikipedia 2018 College football Playoff national championship Jump to : navigation , search 2018 College
Football Playoff National Championship Alabama Crimson Tide Georgia Bulldogs ( 12 -- 1 ) ( 13 -- 1 ) SEC SEC 26 23 Head coach : Nick Saban Head coach : Kirby
Smart AP Coaches CFP AP Coaches CFP OT Total Alabama 0 0 10 10 6 26 Georgia 0 13 7 0 23 Date January 8 , 2018 Season 2017 Stadium Mercedes - Benz Stadium
Location Atlanta , Georgia MVP Offensive : # 13 QB Tua Tagovailoa , Fr . Alabama Defensive : # 94 DT Da'Ron Payne , Jr .
</item 9>

<item 10>
2018 College Football Playoff National Championship
champions Mythical national championship Plus - One system Power Five conferences Championship games for each season are played in January , while semifinal
games are identified by season year , alternating between December and January Links to related articles 2017 -- 18 NCAA football bowl game season New Mexico (
Dec. 16 ) Las Vegas ( Dec. 16 ) Camellia ( Dec. 16 ) Cure ( Dec. 16 ) New Orleans ( Dec. 16 ) Boca Raton ( Dec. 19 ) Frisco ( Dec. 20 ) Gasparilla ( Dec. 21 )
Famous Idaho Potato ( Dec. 22 ) Bahamas ( Dec. 22 ) Armed Forces ( Dec. 23 ) Birmingham ( Dec. 23 ) Dollar General ( Dec. 23 ) Hawaii ( Dec. 24 ) Quick Lane (
Dec. 26 ) Heart of Dallas ( Dec. 26 ) Cactus ( Dec. 26 ) Independence ( Dec. 27 ) Pinstripe ( Dec. 27 ) Texas ( Dec. 27 ) Foster Farms ( Dec. 27 ) Military (
Dec. 28 ) Camping World ( Dec. 28 ) Alamo ( Dec.
</item 10>
=== END ITEM id=37d81753f25bc627 ===

=== ITEM id=b8b91b64d83bf9cd ===
Gold sentences and facts:
Sentence 0: Islami Somaz Bank Limited Al - Arafah Islami Bank Limited EXIM Bank Limited First Security Islami Bank Limited ICB Islamic Bank Limited Islami Bank Bangladesh Limited Shahjalal Islami Bank Limited Social Islami Bank Limited Union Bank Limited
  Fact 0.0: Islami Somaz Bank Limited is listed as a bank.
  Fact 0.1: Al - Arafah Islami Bank Limited is listed as a bank.
  Fact 0.2: EXIM Bank Limited is listed as a bank.
  Fact 0.3: First Security Islami Bank Limited is listed as a bank.
  Fact 0.4: ICB Islamic Bank Limited is listed as a bank.
  Fact 0.5: Islami Bank Bangladesh Limited is listed as a bank.
  Fact 0.6: Shahjalal Islami Bank Limited is listed as a bank.
  Fact 0.7: Social Islami Bank Limited is listed as a bank.
  Fact 0.8: Union Bank Limited is listed as a bank.

Retrieved items:
<item 1>
List of banks in Bangladesh
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
Blood type distribution by country
0 % 0.4 % Australia 24,642,693 40.0 % 31.0 % 8.0 % 2.0 % 9.0 % 7.0 % 2.0 % 1.0 % Austria 8,592,470 30.0 % 33.0 % 12.0 % 6.0 % 7.0 % 8.0 % 3.0 % 1.0 % Bahrain
1,418,695 48.48 % 19.35 % 22.61 % 3.67 % 3.27 % 1.33 % 1.04 % 0.25 % Bangladesh 164,833,667 31.18 % 21.44 % 34.58 % 8.85 % 1.39 % 0.96 % 0.96 % 0.64 % Belgium
11,444,053 38.0 % 34.0 % 8.6 % 4.1 % 7.0 % 6.0 % 1.5 % 0.8 % Bolivia 11,053,
</item 2>

<item 3>
List of banks in Bangladesh
A Commercial Bank of Ceylon PLC Habib Bank Limited National Bank of Pakistan Standard Chartered Bank State Bank of India Punjab National Bank Woori Bank HSBC
Non-scheduled banks ( edit ) Non-Scheduled Banks are licensed only for specific functions and objectives , and do not offer the same range of services as
scheduled banks . Ansar VDP Unnayan Bank Karmashangosthan Bank Probashi Kollyan Bank Grameen Bank Jubilee Bank Palli Sanchay Bank Non Bank Financial
Institutions ( edit ) Investment Corporation of Bangladesh Agrani SME Financing Company Limited Bangladesh Finance and Investment Company Limited Bangladesh
Industrial Finance Company Limited Bangladesh Infrastructure Finance Fund Limited Bay Leasing and Investment Limited CAPM Venture Capital and Finance Limited
Delta Brac Housing Finance Corporation Limited Far - east Finance & Investment Limited FAS Finance & Investment Limited First Finance Limited GSP Finance
Company ( Bangladesh ) Limited Hajj Finance Company Limited IDLC Finance Limited Industrial and Infrastructure Development Finance Company Limited IPDC Finance
Limited Infrastructure Development Company Limited International Leasing and Financial Services Limited Islamic Finance and Investment Limited LankaBangla
Finance Limited Meridian Finance and Investment Limited MIDAS Financing Limited National Finance Limited National Housing Finance and Investments Limited People
's Leasing and Financial Services Limited Phoenix Finance
</item 3>

<item 4>
Economy of India
The Indian money market is classified into the organised sector , comprising private , public and foreign - owned commercial banks and cooperative banks ,
together known as ' scheduled banks ' ; and the unorganised sector , which includes individual or family - owned indigenous bankers or money lenders and
non-banking financial companies . The unorganised sector and microcredit are preferred over traditional banks in rural and sub-urban areas , especially for
non-productive purposes such as short - term loans for ceremonies . Prime Minister Indira Gandhi nationalised 14 banks in 1969 , followed by six others in 1980
, and made it mandatory for banks to provide 40 % of their net credit to priority sectors including agriculture , small - scale industry , retail trade and
small business , to ensure that the banks fulfilled their social and developmental goals . Since then , the number of bank branches has increased from 8,260 in
1969 to 72,170 in 2007 and the population covered by a branch decreased from 63,800 to 15,000 during the same period .
</item 4>

<item 5>
List of banks in Bangladesh
National Finance Limited National Housing Finance and Investments Limited People 's Leasing and Financial Services Limited Phoenix Finance and Investments
Limited Premier Leasing & Finance Limited Prime Finance & Investment Limited Reliance Finance Limited Saudi - Bangladesh Industrial & Agricultural Investment
Company Limited SEAF Bangladesh Venture Limited The UAE - Bangladesh Investment Company Limited Union Capital Limited United Finance Limited Uttara Finance and
Investments Limited Specialized Financial Institutions ( Semi Formal Sector ) ( edit ) House Building Financial Corporation ( HBFC ) Palli Karma Sahayak
Foundation ( PKSF ) References ( edit ) ^ Jump up to : `` Financial System '' . Central Bank of Bangladesh . Retrieved 9 August 2016 . Jump up ^ `` Relevant
links '' . Central Bank of Bangladesh . Retrieved 7 February 2018 .
</item 5>

<item 6>
Economy of India
The total bank deposits increased from ₹ 59.1 billion ( US $910 million ) in 1970 -- 71 to ₹ 38,309.22 billion ( US $590 billion ) in 2008 -- 09 . Despite an
increase of rural branches -- from 1,860 or 22 % of the total in 1969 to 30,590 or 42 % in 2007 -- only 32,270 of 500,000 villages are served by a scheduled
bank . India 's gross domestic savings in 2006 -- 07 as a percentage of GDP stood at a high 32.8 % . More than half of personal savings are invested in physical
assets such as land , houses , cattle , and gold . The government - owned public - sector banks hold over 75 % of total assets of the banking industry , with
the private and foreign banks holding 18.2 % and 6.5 % respectively . Since liberalisation , the government has approved significant banking reforms .
</item 6>

<item 7>
List of banks in Bangladesh
List of banks in Bangladesh - wikipedia List of banks in Bangladesh Jump to : navigation , search The banking sector in Bangladesh consists of several types of
institution : Contents ( hide ) 1 Central bank 2 Scheduled Banks 3 State Owned Banks 4 Private Commercial Banks 5 Islamic Commercial Private Banks 6 Foreign
Commercial Banks 7 Non-Scheduled Banks 8 Non Bank Financial Institutions 9 Specialized Financial Institutions ( Semi Formal Sector ) 10 References Central Bank
( edit ) Bangladesh Bank is the central bank of Bangladesh and the chief regulatory authority in the banking sector . Pursuant to Bangladesh Bank Order , 1972
the Government of Bangladesh reorganized the Dhaka Branch of the State Bank of Pakistan as the central bank of the country , and named it Bangladesh Bank with
retrospective effect from 16 December 1971 . Scheduled banks ( edit ) Secheduled Banks are licensed under the Bank Company Act , 1991 ( Amended up to 2013 ) .
State Owned Banks ( edit ) Sonali Bank Ltd . Janata Bank Ltd . Agrani Bank Ltd . Rupali Bank Ltd . Bangladesh Krishi Bank Bangladesh Development Bank Limited .
</item 7>

<item 8>
Agriculture in India
Reserve Bank of India Governors India Government Mint Financial services Banking Banks Insurance Bombay Stock Exchange National Stock Exchange Multi Commodity
Exchange Bullion Black money Stock exchanges History Economic development Economic liberalisation Licence Raj Green revolution Government initiatives Numbering
system People Billionaires Businesspeople Demography Income Poverty Labour law Pensions EPFO NPS PPF States Andhra Pradesh Assam Bihar Delhi Goa Gujarat Haryana
Himachal Pradesh Jammu and Kashmir Karnataka Kerala Madhya Pradesh Maharashtra Mizoram Odisha Punjab Rajasthan Tamil Nadu Telangana Uttarakhand Uttar Pradesh
West Bengal Sectors Agriculture Livestock Fishing Automotive Defence Construction Education Energy Nuclear Solar Wind Entertainment Forestry Gambling Healthcare
Information technology Media Cinema Television Printing Mining Coal Iron and Steel Pharmaceuticals Retail Science and technology Biotechnology
Telecommunications Textiles Tourism Transport Aviation Civil Ports Rail Utilities Electricity Water Commons Wikiquotes Agriculture in Asia Sovereign states
Afghanistan Armenia Azerbaijan Bahrain Bangladesh Bhutan Brunei Cambodia China Cyprus East Timor ( Timor - Leste ) Egypt Georgia India Indonesia Iran Iraq
Israel Japan Jordan Kazakhstan North Korea South Korea Kuwait Kyrgyzstan Laos Lebanon Malaysia Maldives Mongolia Myanmar Nepal Oman Pakistan Philippines Qatar
Russia Saudi Arabia Singapore Sri Lanka Syria Tajikistan Thailand Turkey Turkmenistan United Arab Emirates Uzbekistan Vietnam Yemen States with limited
recognition Abkhazia
</item 8>

<item 9>
List of banks in Bangladesh
Rupali Bank Ltd . Bangladesh Krishi Bank Bangladesh Development Bank Limited . Basic Bank Limited Private Commercial Banks ( edit ) AB Bank Limited Bangladesh
Commerce Bank Limited Bank Asia Limited Bengal Bank Limited BRAC Bank Limited City Bank Limited Dhaka Bank Limited Dutch - Bangla Bank Limited Eastern Bank
Limited IFIC Bank Limited Jamuna Bank Limited Meghna Bank Limited Mercantile Bank Limited Midland Bank Modhumoti Bank Limited Mutual Trust Bank Limited National
Bank Limited National Credit & Commerce Bank Limited NRB Bank Limited NRB Commercial Bank Limited One Bank Limited Premier Bank Limited Prime Bank Limited
Pubali Bank Limited South Bangla Agriculture & Commerce Bank Limited Southeast Bank Limited Standard Bank Limited The Farmers Bank Limited Trust Bank Limited
United Commercial Bank Limited Uttara Bank Limited Shimanto Bank Ltd Islamic Commercial Private Banks ( edit ) Islami Somaz Bank Limited Al - Arafah Islami Bank
Limited EXIM Bank Limited First Security Islami Bank Limited ICB Islamic Bank Limited Islami Bank Bangladesh Limited Shahjalal Islami Bank Limited Social Islami
Bank Limited Union Bank Limited Foreign Commercial Banks ( edit ) Bank Al - Falah Limited Citibank N.
</item 9>

<item 10>
List of banks in Bangladesh
Central Bank of Bangladesh . Retrieved 7 February 2018 . ( hide ) List of banks in Asia Sovereign states Afghanistan Armenia Azerbaijan Bahrain Bangladesh
Bhutan Brunei Cambodia China Cyprus East Timor ( Timor - Leste ) Egypt Georgia India Indonesia Iran Iraq Israel Japan Jordan Kazakhstan North Korea South Korea
Kuwait Kyrgyzstan Laos Lebanon Malaysia Maldives Mongolia Myanmar Nepal Oman Pakistan Philippines Qatar Russia Saudi Arabia Singapore Sri Lanka Syria Tajikistan
Thailand Turkey Turkmenistan United Arab Emirates Uzbekistan Vietnam Yemen States with limited recognition Abkhazia Artsakh Northern Cyprus Palestine South
Ossetia Taiwan Dependencies and other territories British Indian Ocean Territory Christmas Island Cocos ( Keeling ) Islands Hong Kong Macau Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_banks_in_Bangladesh&oldid=832850714 '' Categories : Lists of companies of Bangladesh Lists of banks by
country Banks of Bangladesh Lists of banks Talk Contents About Wikipedia বাংলা Edit links This page was last edited on 28 March 2018 , at 10 : 02 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 10>
=== END ITEM id=b8b91b64d83bf9cd ===
