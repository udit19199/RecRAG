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

=== ITEM id=4ef84c416a2dc7c6 ===
Gold sentences and facts:
Sentence 0: Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa . Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order , an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star Wars : The Last Jedi ( 2017 ) .
  Fact 0.0: Kylo Ren is a fictional character in the Star Wars franchise.
  Fact 0.1: Kylo Ren was introduced in the 2015 film Star Wars: The Force Awakens.
  Fact 0.2: Kylo Ren is portrayed by Adam Driver.
  Fact 0.3: Kylo Ren is the chosen name of Ben Solo.
  Fact 0.4: Ben Solo is the son of original Star Wars trilogy characters Han Solo and Leia Organa.
  Fact 0.5: Kylo Ren was trained by his uncle Luke Skywalker as a Jedi.
  Fact 0.6: Kylo Ren has been seduced to the dark side of the Force by Supreme Leader Snoke.
  Fact 0.7: Kylo Ren aspires to be as powerful as his grandfather, Darth Vader.
  Fact 0.8: Kylo Ren is the master of the Knights of Ren.
  Fact 0.9: Kylo Ren is a commander of the First Order.
  Fact 0.10: Kylo Ren later became the supreme leader of the First Order.
  Fact 0.11: The First Order is an organization spawned from the fallen Galactic Empire.
  Fact 0.12: Kylo Ren is featured in The Force Awakens media and merchandising.
  Fact 0.13: Kylo Ren appears in the film's sequel, Star Wars: The Last Jedi.
  Fact 0.14: Star Wars: The Last Jedi was released in 2017.
Sentence 1: Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous . Melissa Leon of The Daily Beast describes Ren 's use of the Force as `` formidable '' , citing his ability to stop a blaster shot mid-air , immobilize victims and probe their minds against their will .
  Fact 1.0: Abrams told Empire in August 2015 that Kylo Ren is not a Sith.
  Fact 1.1: Abrams told Empire in August 2015 that Kylo Ren works under Supreme Leader Snoke.
  Fact 1.2: Abrams told Empire in August 2015 that Supreme Leader Snoke is a powerful figure on the Dark Side of the Force.
  Fact 1.3: Abrams had previously told Entertainment Weekly that the character came to the name Kylo Ren when he joined a group called the Knights of Ren.
  Fact 1.4: Robbie Collin of The Telegraph described Ren as a hot-headed, radicalised Dark Side jihadi.
  Fact 1.5: Robbie Collin of The Telegraph described Ren's red lightsaber as spluttering and crackling as violently as his temper.
  Fact 1.6: Abrams noted that the lightsaber is something Ren built himself.
  Fact 1.7: Abrams noted that the lightsaber is as dangerous, fierce and ragged as the character.
  Fact 1.8: The Telegraph explains that Ren's wild and erratic temper and angsty instability make him dangerous.
  Fact 1.9: Melissa Leon of The Daily Beast describes Ren's use of the Force as formidable.
  Fact 1.10: Melissa Leon cites Ren's ability to stop a blaster shot mid-air.
  Fact 1.11: Melissa Leon cites Ren's ability to immobilize victims.
  Fact 1.12: Melissa Leon cites Ren's ability to probe victims' minds against their will.

Retrieved items:
<item 1>
Kylo Ren
Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful
as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order ,
an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star
Wars : The Last Jedi ( 2017 ) . Contents ( hide ) 1 Concept and creation 2 Character 3 Appearances 3.1 Feature films 3.1. 1 The Force Awakens ( 2015 ) 3.1. 2
The Last Jedi ( 2017 ) 3.2 Related works and promotional material 3.3 Novels 4 Reception 5 Family tree 6 References 7 External links Concept and creation ( edit
) After Lucasfilm was acquired by The Walt Disney Company in 2012 , the script for The Force Awakens was written by Lawrence Kasdan , J.J.
</item 1>

<item 2>
Kylo Ren
Abrams Michael Arndt Portrayed by Adam Driver Voiced by Adam Driver ( Disney Infinity 3.0 , Lego Star Wars : The Force Awakens , Lego Star Wars : The Resistance
Rises ) Matthew Wood ( Star Wars Battlefront II ) Roger Craig Smith ( Star Wars Battlefront II ) Information Full name Ben Solo Species Human Gender Male
Occupation Supreme Leader of the First Order Master of the Knights of Ren Former Jedi Padawan Affiliation First Order Knights of Ren New Jedi Order ( formerly )
Family Han Solo ( father ) Leia Organa ( mother ) Relatives Luke Skywalker ( maternal uncle ) Anakin Skywalker ( maternal grandfather ) Padmé Amidala ( maternal
grandmother ) Bail Organa ( adopted maternal grandfather ) Breha Organa ( adopted maternal grandmother ) Shmi Skywalker ( maternal great - grandmother )
Homeworld Chandrila Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed
by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa .
</item 2>

<item 3>
Kylo Ren
Novels ( edit ) Kylo Ren is a point of view character in the 2015 novelization of The Force Awakens by Alan Dean Foster . In the 2017 Chuck Wendig novel Star
Wars : Aftermath : Empire 's End , Ben Solo is stated as having been born on the planet Chandrila on the same day as a peace treaty is signed between the
remnants of the Empire and the New Galactic Republic . Kylo Ren is additionally mentioned in the 2017 novel Star Wars : Phasma . Reception ( edit ) Adam Driver
received positive reviews for his performance as Kylo Ren . Both the character of Kylo Ren and Driver 's performance have been positively received . Many
complimented Ren 's conflicted nature and depth , as well as his costume design , and noted there were many places the character could be taken in future
installments . Peter Bradshaw of The Guardian praised the character and the actor alike , saying , `` He is gorgeously cruel , spiteful and capricious -- and
unlike the Vader of old , he is given to petulant temper tantrums , with his lightsaber drawn . ''
</item 3>

<item 4>
Kylo Ren
Driver won a 2016 Saturn Award for Best Supporting Actor for his portrayal . In January 2018 , Kylo was voted seventh greatest movie villain of all time by the
readers of Empire . Family tree ( edit ) Main article : Solo family ( hide ) Skywalker family tree Aika Lars Cliegg Lars Shmi Skywalker Jobal Naberrie Ruwee
Naberrie Beru Whitesun Owen Lars Anakin Skywalker Padmé Amidala Bail Organa Breha Organa Luke Skywalker Leia Organa Han Solo Ben Solo References : Jump up ^
Hidalgo , Pablo ( 2016 ) . Star Wars Character Encyclopedia : Updated and Expanded . Jump up ^ `` Databank : Cliegg Lars '' . StarWars.com . Retrieved December
3 , 2015 . Jump up ^ `` Databank : Shmi Skywalker Lars '' . StarWars.com . Retrieved December 3 , 2015 . ^ Jump up to : Attack of the Clones ( 2002 ) and
Revenge of the Sith ( 2005 ) Jump up ^ `` Databank : Beru Lars '' . StarWars.com .
</item 4>

<item 5>
Kylo Ren
Rey finds herself able to experience Ren 's emotions and confronts him on his fear that he will never be as powerful as Vader . She later uses the `` Jedi mind
trick '' to compel her stormtrooper guard to let her escape . Han arrives at the First Order 's battle station , Starkiller Base , as part of the Resistance 's
plan to destroy it . After Han plants explosives to destroy the base , he confronts Ren , calling him by his real name , Ben , and implores him to abandon the
dark side , warning him that Snoke will kill him once he accomplishes his goal to seize control of the galaxy . Ren refuses and kills his father , enraging
Chewbacca , Han 's copilot , who fires , injuring Ren . As Finn and Rey flee the damaged base , a badly injured Ren follows and confronts them . Finn fights Ren
with Luke 's recovered lightsaber , but Ren overpowers and severely wounds him . Rey then takes up the lightsaber and , using the Force , begins to overcome Ren
, as well as striking him on the face with the lightsaber , leaving Ren with a scar on the right side of his face .
</item 5>

<item 6>
Kylo Ren
Ren 's character arc shares similarities with that of the Star Wars expanded universe character Jacen Solo , the son of Han Solo and Princess Leia who threatens
the galaxy as a fallen Jedi . Additionally , critics have noted a resemblance between Ren 's character design and that of Revan , the protagonist of Knights of
the Old Republic . Character ( edit ) Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful
figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a
group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber
splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce
and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous .
</item 6>

<item 7>
Kylo Ren
StarWars.com . Retrieved December 4 , 2015 . Jump up ^ In the alternate Star Wars expanded universe ( Legends ) , Luke is married to Mara Jade and has a son ,
Ben Skywalker . Jump up ^ `` Databank : Princess Leia Organa '' . StarWars.com . Retrieved December 4 , 2015 . ^ Jump up to : The Force Awakens. 2015 . Jump up
^ In the alternate Star Wars expanded universe ( Legends ) continuity , Han and Leia have three children : Jaina , Jacen and Anakin Solo . References ( edit )
Jump up ^ Wendig , Chuck ( 2017 ) . Star Wars : Aftermath : Empire 's End . Del Rey . ISBN 9781101966976 . What is known is this : The child 's name is Ben ,
and he takes his father 's last name , even as Leia keeps only her own family name , Organa . Jump up ^ Star Wars : The Force Awakens ( Little Golden Book ) .
Golden Books . April 12 , 2016 . ISBN 9780736434911 .
</item 7>

<item 8>
Kylo Ren
com Databank Kylo Ren on Wookieepedia , a Star Wars wiki Kylo Ren on IMDb Star Wars characters List of The Clone Wars characters List of Rebels characters List
of Legends characters Characters Canon Admiral Ackbar Padmé Amidala Cassian Andor Wedge Antilles Cad Bane Darth Bane BB - 8 Jar Jar Binks C - 3PO Lando
Calrissian Chewbacca Lieutenant Connix Poe Dameron Count Dooku Jyn Erso Boba Fett Jango Fett Finn Saw Gerrera Greedo Tag Greenley General Grievous Jabba the
Hutt General Hux Kanan Jarrus Qui - Gon Jinn K - 2SO Maz Kanata Obi - Wan Kenobi Orson Krennic Princess Leia Darth Maul Nien Nunb Bink Otauna Palpatine Captain
Phasma Admiral Piett R2 - D2 Kylo Ren Rey Captain Rex Bodhi Rook Bail Organa Luke Skywalker Supreme Leader Snoke Han Solo Ahsoka Tano Grand Moff Tarkin Grand
Admiral Thrawn Rose Tico Darth Vader Asajj Ventress Iden Versio Watto Wicket W.
</item 8>

<item 9>
Kylo Ren
Kylo Ren - wikipedia Kylo Ren Jump to : navigation , search `` Kylo '' redirects here . For the web browser , see Kylo ( web browser ) . Kylo Ren Ben Solo Star
Wars character Kylo Ren 's costume displayed on a mannequin First appearance The Force Awakens ( 2015 ) Created by Lawrence Kasdan J.J.
</item 9>

<item 10>
Kylo Ren
You normally think of order , and structure , and full commitment and no hesitation ... he 's just a little bit more unpolished . It 's in his costume , in his
lightsaber -- how you kind of get the sense that it could just not work at any moment ; that it could just blow up . That 's kind of like a big metaphor for him
. Appearances ( edit ) Model of Kylo 's cross-bladed lightsaber on display at Star Wars Launch Bay at Disney 's Hollywood Studios ( 2015 ) . Feature films (
edit ) The Force Awakens ( 2015 ) ( edit ) Main article : Star Wars : The Force Awakens Kylo Ren first appears in The Force Awakens as the commander of the
First Order , a sinister regime that has risen from the remains of the old Galactic Empire . After arriving at Jakku to retrieve a map containing the
coordinates for the location of Luke Skywalker , Ren kills an old vicar known as Lor San Tekka and captures Resistance pilot Poe Dameron , who has also been
sent to recover the map by General Leia Organa .
</item 10>
=== END ITEM id=4ef84c416a2dc7c6 ===

=== ITEM id=804250f4083a5333 ===
Gold sentences and facts:
Sentence 0: Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ; sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
  Fact 0.0: Lynne is the eldest of the Slater sisters.
  Fact 0.1: Lynne arrives in Albert Square in 2000.
  Fact 0.2: Lynne arrives in Albert Square with her family.
  Fact 0.3: Lynne's father is Charlie, played by Derek Martin.
  Fact 0.4: Lynne's sister Kat is played by Jessie Wallace.
  Fact 0.5: Lynne's sister Little Mo is played by Kacey Ainsworth.
  Fact 0.6: Lynne's sister Zoe is played by Michelle Ryan.
  Fact 0.7: Lynne's grandmother is Mo, played by Laila Morse.
  Fact 0.8: Unlike Little Mo and Zoe, Lynne is aware that Zoe is in fact Kat's daughter.
  Fact 0.9: Lynne helps keep the family secret that Zoe is Kat's daughter until Kat reveals the truth to Zoe.

Retrieved items:
<item 1>
Lynne Hobbs
In January 2016 , Lynne attends Charlie 's funeral off - screen . Creation and development ( edit ) In 2000 , EastEnders ' executive producer John Yorke decided
to introduce the `` classic '' Slater family . He felt the show needed to go back to its roots and bring back some traditional values . BBC 's head of drama ,
Mal Young commented , `` We do not have enough solid families in the soap , there were a lot of fractured families and people who were alone . '' The family
were created as a replacement for the di Marco family , who were axed by Yorke . The family consisted of , grandmother Mo Harris ( Laila Morse ) , father
Charlie ( Derek Martin ) , his four children , Lynne ( Elaine Lordan ) , Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) , Zoe ( Michelle Ryan ) , as
well as Lynne 's boyfriend Garry Hobbs ( Ricky Groves ) . It later transpired that Kat was the mother of Zoe , after being raped by her uncle Harry Slater (
Michael Elphick ) .
</item 1>

<item 2>
Lynne Hobbs
Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe Wicks Kevin Wicks Lorraine Wicks Simon Wicks Debbie Wilkins Cindy Williams
James Willmott - Brown Willy May Wright Lists of characters By year : 1985 1986 2017 2018 Others : Flaherty family Dickens Hill inmates The Banned band members
Characters from spin - offs Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lynne_Hobbs&oldid=838104750 '' Categories : EastEnders characters
Fictional characters introduced in 2000 Female characters in television Hidden categories : Articles needing additional references from March 2011 All articles
needing additional references All articles with unsourced statements Articles with unsourced statements from January 2011 Use dmy dates from March 2011 Talk
Contents About Wikipedia Gaeilge Edit links This page was last edited on 24 April 2018 , at 23 : 33 . About Wikipedia
</item 2>

<item 3>
Lynne Hobbs
links ( edit ) Lynne Hobbs at BBC Online EastEnders characters Present and future characters Kathy Beale Sharon Watts Ian Beale Dot Cotton Martin Fowler Phil
Mitchell Sonia Fowler Robbie Jackson Mel Owen Billy Mitchell Mo Harris Kat Slater Patrick Trueman Louise Mitchell Bobby Beale Stacey Slater Jean Slater Honey
Mitchell Denise Fox Max Branning Janet Mitchell Shirley Carter Jay Brown Rainie Cross Masood Ahmed Jack Branning Whitney Dean Tiffany Butcher Kim Fox Cora Cross
Tina Carter Linda Carter Mick Carter Charlie Cotton ( hide ) Past characters AJ Ahmed Della Alexander Johnny Allen Ruby Allen Aunt Sal Tom Banks Geoff Barnes
Clare Bates Debbie Bates Laurie Bates Nigel Bates Cindy Beale Jane Beale Laura Beale Lou Beale Lucy Beale Pete Beale Peter Beale Steven Beale Polly Becker Adam
Best Manda Best Abi Branning Alice Branning April Branning Bradley Branning Derek Branning Jim Branning Lauren Branning Suzy Branning Tanya Branning Aidan
Brosnan Diane Butcher Frank Butcher Janine Butcher Liam Butcher Mo Butcher Pat Butcher Ricky Butcher Kelvin Carpenter Tony Carpenter Nancy Carter Stan Carter
Zsa Zsa Carter Lorna Cartwright Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986
</item 3>

<item 4>
Lynne Hobbs
Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986 ) Dotty Cotton Nick Cotton Stella Crawford Beppe di Marco
Bruno and Luisa di Marco Gianni di Marco Rosa di Marco Ray Dixon Jase Dyer Huw Edwards Steve Elliot Nellie Ellis Barry Evans Natalie Evans Roy Evans Fatboy
Ferreira family Conor Flaherty Mary Flaherty Fred Fonseca Arthur Fowler Lisa Fowler Mark Fowler Michelle Fowler Pauline Fowler Ruth Fowler Vicki Fowler Chelsea
Fox Libby Fox Joanne Francis Jodie Gold Vanessa Gold Marge Green Nina Harris Dexter Hartman Julie Haye Alex Healy Christine Hewitt Ted Hills Tony Hills Garry
Hobbs Lynne Hobbs Lofty Holloway Vincent Hubbard Andy Hunter Alan Jackson Bianca Jackson Billie Jackson Blossom Jackson Carmel Jackson Carol Jackson Naima
Jeffery Saeed Jeffery Al Jenkins Greg Jessop Lucas Johnson Danielle Jones Naomi Julien Gita Kapoor Sanjay Kapoor Karim family Yusef Khan Tony King Harold Legg
Donna Ludlow Joe Macer Ryan Malloy Jill Marsden Afia Masood Amira Masood Shabnam Masood Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie
</item 4>

<item 5>
List of The Fall characters
Katie is a gifted musician and uses online videos to express her feelings . During series 2 she becomes deeply embroiled in Paul 's life . Paul ties her up in a
hotel room , for example , whilst Katie breaks into a house wearing a shirt with Spector 's face on . Benedetto is placed under police surveillance and later
arrested whilst trying to dispose of evidence that implicates Paul in the murders . In series three , she attacks a girl with acid , and is detained for her
crimes . During this time , she reveals to Stella that her father 's death had a severe emotional impact on her , and she has been self - harming . Sally Ann
Spector ( edit ) Sally Ann Spector ( Bronagh Waugh ) , the wife of Paul Spector and a neonatal nurse . She has a deep devotion to Paul , but is wary of his
character as she has never met his family . Sally Ann is the mother of his two children and is unaware of his dark side . Over the course of the series , she
becomes embroiled in his crimes without her knowledge , and even provides a false alibi to the police .
</item 5>

<item 6>
Lynne Hobbs
2000 -- 2004 ( edit ) Lynne is in a relationship with hapless mechanic Garry Hobbs ( Ricky Groves ) and spends much of her time despairing over Garry 's
inability to fully commit to their relationship . All Lynne really wants is to marry Garry , settle down and raise a family but while Garry loves Lynne , he is
reluctant to lose his `` bachelor status '' and he spends much of his time dodging Lynne 's ' hints ' about marriage . However , Lynne ( and the rest of the
Slaters ) wear Garry down and the two eventually get engaged . Lynne works in Ian Beale 's ( Adam Woodyatt ) café and often clashes with co-worker , Janine
Butcher ( Charlie Brooks ) , who regularly slacks off work and flirts with Garry to irritate her . This culminates in Lynne eventually getting Janine fired and
the two become enemies thereafter . In 2001 , Lynne becomes friends with womanising club owner Beppe di Marco ( Michael Greco ) and she often babysits his son ,
Joe ( Jake Kyprianou ) .
</item 6>

<item 7>
Lynne Hobbs
Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie McFarlane Poppy Meadow Darren Miller Demi Miller Keith Miller Mickey Miller Rosie Miller Rob Minter
Archie Mitchell Ben Mitchell Danny Mitchell Glenda Mitchell Grant Mitchell Jamie Mitchell Kate Mitchell Little Mo Mitchell Peggy Mitchell Ronnie Mitchell Roxy
Mitchell Sam Mitchell Tiffany Mitchell Vinnie Monks Alfie Moon Anthony Moon Danny Moon Eddie Moon Jake Moon Michael Moon Nana Moon Spencer Moon Tyler Moon
Trevor Morgan Rod Norman Andy O'Brien Mercy Olubunmi Ali Osman Mehmet Osman Sue Osman Steve Owen Annie Palmer George Palmer Lola Pearce Minty Peterson Frankie
Pierre Paul Priestly Irene Raymond Louise Raymond Simon Raymond Terry Raymond Dennis Rickman Binnie Roberts Roly Willy Roper Matthew Rose Colin Russell Mandy
Salter Trevor Short Lydia Simmonds Ethel Skinner Charlie Slater Sean Slater Zoe Slater Leon Small Babe Smith Gus Smith Mary Smith Dan Sullivan Gavin Sullivan
Dawn Swann Celestine and Etta Tavernier Clyde Tavernier Hattie Tavernier Jules Tavernier Kelly Taylor Heather Trott Anthony Trueman Paul Trueman Yolande Trueman
Owen Turner Angie Watts Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe
</item 7>

<item 8>
Lynne Hobbs
Lynne is a fierce judge of right and wrong , although she is not always able to practise what she preaches . She has a turbulent relationship with husband Garry
( Ricky Groves ) , but despite her complaints about him , it ends up being Lynne 's adultery that puts their relationship in jeopardy long before Garry strays .
Contents ( hide ) 1 Storylines 1.1 Backstory 1.2 2000 -- 2004 2 Creation and development 3 In popular culture 4 References 5 External links Storylines ( edit )
Backstory ( edit ) Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ;
sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne
is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
</item 8>

<item 9>
Lynne Hobbs
Unsourced material may be challenged and removed . ( March 2011 ) ( Learn how and when to remove this template message ) Lynne Hobbs EastEnders character
Portrayed by Elaine Lordan Duration 2000 -- 2004 First appearance Episode 2003 18 September 2000 ( 2000 - 09 - 18 ) Last appearance Episode 2756 2 July 2004 (
2004 - 07 - 02 ) Created by Tony Jordan Introduced by John Yorke Spin - off appearances Slaters in Detention ( 2003 ) Classification Former ; regular Profile
Other names Lynne Slater Occupation Café worker ( show ) Family Father Charlie Slater Mother Viv Slater Sisters Kat Slater Little Mo Mitchell Belinda Peacock
Husband Garry Hobbs ( 2001 -- 2004 ) Daughters Vivienne Hobbs Grandmothers Mo Harris Uncles Harry Slater Nephews Freddie Mitchell Nieces Zoe Slater Lynne Hobbs
( also Slater ) is a fictional character from the BBC soap opera EastEnders , played by Elaine Lordan from September 2000 to July 2004 .
</item 9>

<item 10>
Lynne Hobbs
In 2003 , another affair jeopardises Lynne 's marriage , but this time it is Garry who strays . Garry has drunken sex with Laura Beale ( Hannah Waterman ) and
she gets pregnant . Laura 's husband , Ian , reveals to her that he has had a vasectomy so the baby can not be his and throws her out so she turns to Garry for
support , thinking he is her baby 's father . Lynne has been trying for a baby and struggles to forgive Garry for having a baby with another woman , throwing
him out so he moves in with Laura to support her during her pregnancy . Lynne finally comes round and forgives Garry after she gets jealous when he has sex with
Lynne 's sister , Belinda Peacock ( Leanne Lakey ) . Lynne and Garry reconcile and in 2004 , Lynne becomes pregnant and a scan confirms that she is expecting a
girl , that they plan to name her Vivienne after Lynne 's late mother , Viv ( April Martin ) . However , disaster strikes when a fairground ride collapses in
Albert Square and Lynne is trapped under the debris .
</item 10>
=== END ITEM id=804250f4083a5333 ===

=== ITEM id=46e55272914c7b65 ===
Gold sentences and facts:
Sentence 0: The earliest predecessors of the eye were photoreceptor proteins that sense light , found even in unicellular organisms , called `` eyespots '' . Eyespots can only sense ambient brightness : they can distinguish light from dark , sufficient for photoperiodism and daily synchronization of circadian rhythms . They are insufficient for vision , as they can not distinguish shapes or determine the direction light is coming from . Eyespots are found in nearly all major animal groups , and are common among unicellular organisms , including euglena . The euglena 's eyespot , called a stigma , is located at its anterior end . It is a small splotch of red pigment which shades a collection of light sensitive crystals . Together with the leading flagellum , the eyespot allows the organism to move in response to light , often toward the light to assist in photosynthesis , and to predict day and night , the primary function of circadian rhythms . Visual pigments are located in the brains of more complex organisms , and are thought to have a role in synchronising spawning with lunar cycles . By detecting the subtle changes in night - time illumination , organisms could synchronise the release of sperm and eggs to maximise the probability of fertilisation .
  Fact 0.0: The earliest predecessors of the eye were photoreceptor proteins that sense light.
  Fact 0.1: The light-sensing photoreceptor proteins that were the earliest predecessors of the eye are found even in unicellular organisms.
  Fact 0.2: These early light-sensing structures are called 'eyespots'.
  Fact 0.3: Eyespots can only sense ambient brightness.
  Fact 0.4: Eyespots can distinguish light from dark.
  Fact 0.5: Sensing ambient brightness is sufficient for photoperiodism.
  Fact 0.6: Sensing ambient brightness is sufficient for daily synchronization of circadian rhythms.
  Fact 0.7: Eyespots are insufficient for vision.
  Fact 0.8: Eyespots cannot distinguish shapes.
  Fact 0.9: Eyespots cannot determine the direction light is coming from.
  Fact 0.10: Eyespots are found in nearly all major animal groups.
  Fact 0.11: Eyespots are common among unicellular organisms.
  Fact 0.12: Euglena is a unicellular organism with an eyespot.
  Fact 0.13: The euglena's eyespot is called a stigma.
  Fact 0.14: The euglena's stigma is located at its anterior end.
  Fact 0.15: The euglena's stigma is a small splotch of red pigment.
  Fact 0.16: The euglena's stigma shades a collection of light-sensitive crystals.
  Fact 0.17: Together with the leading flagellum, the euglena's eyespot allows the organism to move in response to light.
  Fact 0.18: The euglena often moves toward the light to assist in photosynthesis.
  Fact 0.19: The euglena's eyespot allows the organism to predict day and night.
  Fact 0.20: Predicting day and night is the primary function of circadian rhythms.
  Fact 0.21: Visual pigments are located in the brains of more complex organisms.
  Fact 0.22: Visual pigments are thought to have a role in synchronising spawning with lunar cycles.
  Fact 0.23: Organisms could detect subtle changes in night-time illumination.
  Fact 0.24: By detecting night-time illumination changes, organisms could synchronise the release of sperm and eggs.
  Fact 0.25: Synchronising the release of sperm and eggs maximises the probability of fertilisation.

Retrieved items:
<item 1>
Evolution of the eye
`` Eye ancestry : old genes for new eyes '' . Current Biology . 6 ( 1 ) : 39 -- 42 . doi : 10.1016 / S0960 - 9822 ( 02 ) 00417 - 7 . PMID 8805210 . Jump up ^
Zinovieva , R. ; Piatigorsky , J. ; Tomarev , S.I. ( 1999 ) . `` O - Crystallin , arginine kinase and ferritin from the octopus lens '' . Biochimica et
Biophysica Acta ( BBA ) - Protein Structure and Molecular Enzymology. 1431 ( 2 ) : 512 -- 517 . doi : 10.1016 / S0167 - 4838 ( 99 ) 00066 - 7 . ^ Jump up to :
Scotland , R.W. ( 2010 ) . `` Deep homology : A view from systematics '' . BioEssays. 32 ( 5 ) : 438 -- 449 . doi : 10.1002 / bies.
</item 1>

<item 2>
Evolution of the eye
`` How Complexity Originates : The Evolution of Animal Eyes '' . Annual Review of Ecology , Evolution , and Systematics. 46 : 237 -- 260 . doi : 10.1146 /
annurev - ecolsys - 110512 - 135907 . Ed Young ; photographs by David Liittschwager ( February 2016 ) . `` Inside the Eye : Nature 's Most Exquisite Creation ''
. National Geographic . 229 ( 2 ) : 30 -- 57 . External links ( edit ) `` Evolution of the Eye '' . WGBH Educational Foundation and Clear Blue Sky Productions .
PBS . 2001 . Creationism Disproved ? Video from the National Center for Science Education on the evolution of the eye Evolution : Education and Outreach Special
Issue : Evolution and Eyes volume 1 , number 4 , October 2008 , pages 351 - 559 . ISSN 1936 - 6426 ( Print ) 1936 - 6434 ( Online ) Evolutionary biology
Evolutionary history of life Index of evolutionary biology articles Introduction Outline of evolution Timeline of evolution Evolution Abiogenesis Adaptation
Adaptive radiation Cladistics
</item 2>

<item 3>
Evolution of the eye
Jump up ^ Dawkins , Richard ( 1986 ) . The Blind Watchmaker . ^ Jump up to : Schoenemann , B. ; Liu , J.N. ; Shu , D.G. ; Han , J. ; Zhang , Z.F. ( 2008 ) . ``
A miniscule optimized visual system in the Lower Cambrian '' . Lethaia. 42 ( 3 ) : 265 -- 273 . doi : 10.1111 / j. 1502 - 3931.2008. 00138. x . Jump up ^ Ali ,
M.A. and M.A. Klyne. 1985 . Vision in vertebrates . New York : Plenum Press Jump up ^ Fernald , Russell D. ( 2001 ) . The Evolution of Eyes : Why Do We See What
We See ? Karger Gazette 64 : `` The Eye in Focus '' . Jump up ^ Fernald , Russell D. ( 1998 ) . Aquatic Adaptations in Fish Eyes . New York , Springer .
</item 3>

<item 4>
Evolution of the eye
Nilsson and S. Pelger estimated in a classical paper how many generations are needed to evolve a complex eye in vertebrates . Another researcher , G.C. Young ,
has used the fossil record to infer evolutionary conclusions , based on the structure of eye orbits and openings in fossilized skulls for blood vessels and
nerves to go through . All this adds to the growing amount of evidence that supports Darwin 's theory . Rate of evolution ( edit ) The first fossils of eyes
found to date are from the lower Cambrian period ( about 540 million years ago ) . The lower Cambrian had a burst of apparently rapid evolution , called the ``
Cambrian explosion '' . One of the many hypotheses for `` causes '' of the Cambrian explosion is the `` Light Switch '' theory of Andrew Parker : It holds that
the evolution of eyes started an arms race that accelerated evolution . Before the Cambrian explosion , animals may have sensed light , but did not use it for
fast locomotion or navigation by vision . The rate of eye evolution is difficult to estimate , because the fossil record , particularly of the lower Cambrian ,
is poor .
</item 4>

<item 5>
Evolution of the eye
Oxford Journals. 96 ( 3 ) : 171 -- 184 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Retrieved 2008 - 04 - 26 . ^ Jump up to : MF Land ; RD Fernald ( 1992
) . `` The Evolution of Eyes '' . Annual Review of Neuroscience. 15 : 1 -- 29 . doi : 10.1146 / annurev. ne. 15.030192. 000245 . PMID 1575438 . Jump up ^ Autrum
, H ( 1979 ) . `` Introduction '' . In H. Autrum . Comparative Physiology and Evolution of Vision in Invertebrates - A : Invertebrate Photoreceptors . Handbook
of Sensory Physiology . VII / 6A . New York : Springer - Verlag . pp. 6 -- 9 . ISBN 3 - 540 - 08837 - 7 .
</item 5>

<item 6>
Evolution of the eye
149 Jump up ^ Lee , M.S.Y. ; Jago , J.B. ; Garcia - Bellido , D.C. ; Edgecombe , G.E. ; Gehling , J. G ; Paterson , J.R. ( 2011 ) . `` Modern optics in
exceptionally preserved eyes of Early Cambrian arthropods from Australia '' . Nature . 474 : 631 -- 634 . doi : 10.1038 / nature10097 . PMID 21720369 . ^ Jump
up to : Darwin , Charles ( 1859 ) . On the Origin of Species . London : John Murray . Jump up ^ Gehring WJ ( 2005 ) . `` New perspectives on eye development and
the evolution of eyes and photoreceptors '' . J. Hered. 96 ( 3 ) : 171 -- 84 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Jump up ^ Nilsson , D. - E. (
2013 ) . `` Eye evolution and its functional basis '' . Visual Neuroscience.
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
Evolution of the eye
Evolution of the eye - wikipedia Evolution of the eye Jump to : navigation , search Major stages in the evolution of the eye in vertebrates . The evolution of
the eye is attractive to study , because the eye distinctively exemplifies an analogous organ found in many animal forms . Complex , image - forming eyes have
evolved independently several times . The evolutionary geneticist Walter Gering holds a different opinion . According his theory , the genes engaged with
photoreceptor cells ( so called Pax 6 genes ) appeared probably at once in the life history , at a very early stage of the evolution . Probably , the complex ,
image forming kinds of eyes are at least five : Chidaria - lens eye , complex eye ; Molluscs - lens eye , everse retina ; Annelids - lens eye , everse retina ;
Arthropods - facet eye ; Vertebrate - lens eye , inverse retina Complex eyes appeared first within the few million years of the Cambrian explosion . From before
the Cambrian , no evidence of eyes has survived , but diverse eyes are known from the Burgess shale of the Middle Cambrian , and from the slightly older Emu Bay
Shale .
</item 8>

<item 9>
Evolution of the eye
PMID 7892602 . Jump up ^ Tomarev , S.I. ; Callaerts , P. ; Kos , L. ; Zinovieva , R. ; Halder , G. ; Gehring , W. ; Piatigorsky , J. ( 1997 ) . `` Squid Pax - 6
and eye development '' . Proceedings of the National Academy of Sciences of the United States of America . 94 ( 6 ) : 2421 -- 2426 . Bibcode : 1997PNAS ...
94.2421 T . doi : 10.1073 / pnas. 94.6. 2421 . PMC 20103 . PMID 9122210 . ^ Jump up to : Gehring , W.J. ( 13 January 2005 ) . `` New Perspectives on Eye
Development and the Evolution of Eyes and Photoreceptors '' ( Full text ) . Journal of Heredity . Oxford Journals. 96 ( 3 ) : 171 -- 184 .
</item 9>

<item 10>
Evolution of the eye
1 ( 4 ) : 439 -- 447 . doi : 10.1007 / s12052 - 008 - 0084 - 1 . ISSN 1936 - 6426 . Further reading ( edit ) Lamb TD , Collin SP , Pugh EN ( December 2007 ) .
`` Evolution of the vertebrate eye : opsins , photoreceptors , retina and eye cup '' . Nat. Rev. Neurosci. 8 ( 12 ) : 960 -- 76 . doi : 10.1038 / nrn2283 . PMC
3143066 . PMID 18026166 . Illustration . Review Lamb , TD ( 2011 ) . `` Evolution of the Eye '' ( PDF ) . Scientific American . 305 : 64 -- 69 . doi : 10.1038 /
scientificamerican0711 - 64 . Land , Michael F. ; Nilsson , Dan - Eric ( 2012 ) . `` The origin of vision '' . Animal Eyes ( 2 ed . ) . Oxford : Oxford
University Press .
</item 10>
=== END ITEM id=46e55272914c7b65 ===

=== ITEM id=f55cbe62c1d39b92 ===
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
California Highway Patrol
Local police or the local sheriff 's department having a contract with an incorporated city are primarily responsible for investigating and enforcing traffic
laws in incorporated cities , but the CHP can still enforce traffic laws on any public road anywhere in the state . While the agency 's primary mission is
related to transportation , it also possesses full law enforcement authority and can enforce any state law anywhere in the state . Furthermore , CHP officers
act as bailiffs for the California Supreme Court and California Court of Appeal , as well as security at State of California buildings . The Ford Police
Interceptor Utility Vehicle replaced the Ford Crown Victoria Police Interceptor in 2013 . CHP officers enforce the California Vehicle Code , pursue fugitives
spotted on the highways , and attend to all significant obstructions and crashes within their jurisdiction . They patrol in various vehicles including Ford
Crown Victoria Police Interceptors , Dodge Charger ( LX ) s , Chevrolet Camaros , BMW R1150RT ‐ P motorcycles , Cessna 206 airplanes , and helicopters which
include Bell OH ‐ 58As , Bell 206L ‐ IVs and Eurocopter AS ‐ 350B ‐ 3s .
</item 1>

<item 2>
California Highway Patrol
Contents ( hide ) 1 Highway patrol duties 2 Special responsibilities 3 Pay and pensions 4 Organization 4.1 Hierarchy 4.2 Field Organization 5 Rank structure 6
Traditions 7 Seven points of the CHP badge 8 Code of honor 9 Fallen officers 10 Mexico Liaison Unit 11 Scandals 11.1 Officer Peyer Scandal 11.2 Newhall incident
12 Mergers 13 Weapons 14 Vehicles 14.1 Origins 14.2 Motorcycles 14.3 Vehicles 14.4 Aircraft 15 Rogue officers 15.1 George Gwaltney 15.2 Craig Peyer 16 Media
references 17 Programs 17.1 Youth programs 17.2 Vehicle Theft Recognition and Awards Program 17.3 Other Programs 18 Origins of the California Highway Patrol 's
name 19 See also 20 References 20.1 Further reading 21 External links Highway Patrol duties ( edit ) The agency has jurisdiction over all California state
routes ( including all freeways and expressways ) , U.S. Highways , Interstate Highways and all public roads in unincorporated and incorporated parts of a
county .
</item 2>

<item 3>
California Highway Patrol
Cold weather and utility uniforms are dark blue . California Highway Patrol campaign hat . Leather gear is black with brass snaps and buckles . Standard traffic
enforcement patrol vehicles are required by state law to have a white door with , in the case of the CHP , a star . The CHP operates traditional black and white
as well as all - white patrol vehicles . The California Highway Patrol is one of the few organizations to continue to use the older toll - free `` Zenith 1 ‐
2000 '' number . With the falling cost of telephone area code 800 and 888 numbers , most organizations have chosen to switch to one of the newer numbers and
discontinue use of the Zenith service , which requires operator assistance . The CHP 's traditions include its own radio codes , widely adopted by local
agencies . The most important is 11 ‐ 99 , which signifies that an officer needs emergency assistance or that an officer is down . In 1981 a charitable
foundation called the 11 ‐ 99 Foundation was founded to provide benefits and scholarships to officers and their families .
</item 3>

<item 4>
California Highway Patrol
Early motorcycles used included Indian , Harley - Davidson and Henderson manufactured bikes during the 1920s and 30s ; though by 1941 the main manufacturers
used at the training academy were Harley - Davidson and Indian . In the 1970s CHP decided to renew their Harleys fleet and tested many motorcycles such as Honda
750 four , Suzuki 850 , Moto Guzzi and finally Kawasaki Z900 and then 1000C1 made famous in CHiPs . In 1998 CHP began using the BMW R1100RT - P , with features
including a helmet interface for communications , two batteries , and a top speed of 130 miles per hour ( 210 km / h ) . Vehicles ( edit ) California Highway
Patrol cruiser on display at Public Safety Day in Lakewood , California . The Department of General Services is responsible for selecting a vehicle based on
price , performance and payload capacity for the CHP . The bid specifications require a pursuit - rated , rear - wheel drive or all - wheel drive that can carry
at least 1,500 pounds , the approximate weight of four officers , their equipment , and the police vehicle equipment .
</item 4>

<item 5>
California Highway Patrol
Beginning in 2013 , the CHP switched to the Ford Explorer - based Ford Police Interceptor Utility fleet vehicles because the Ford Crown Victoria Police
Interceptor was being phased out of production . CHP officers are responsible for investigating and disposing of car accidents , disabled vehicles , debris ,
and other impediments to the free flow of traffic . They are often the first responders at the scene of an accident ( or obstruction ) , and in turn summon
paramedics , firefighters , tow truck drivers or Caltrans personnel . The CHP files traffic collision reports for state highways and within unincorporated areas
. The CHP responds to and investigates all accidents involving school buses throughout the state including incorporated cities . CHP also has Multidisciplinary
Accident Investigation Teams ( MAIT ) for the investigation of traffic collisions . Special responsibilities ( edit ) The CHP also publishes data on traffic
accidents in California from a database called SWITRS ( Statewide Integrated Traffic Records System ) . Members of the CHP SWAT Team After the September 11 ,
2001 attacks the CHP became responsible for securing and patrolling a number of potential terrorist targets in California .
</item 5>

<item 6>
California Highway Patrol
Legal jurisdiction As per operations jurisdiction Governing body California State Legislature Constituting instrument Senate Bill 869 Headquarters Sacramento ,
California Officers 7,608 ( as of 2016 ) Civilians 3,276 ( as of 2014 ) Commissioner responsible Warren Stanley Parent agency California State Transportation
Agency ( CALSTA ) Facilities Commands 8 Field Divisions 16 Commercial Enforcement Facilities 103 Area Offices Patrol cars Ford Explorer - based Police
Interceptor Utility Vehicle Dodge Charger - Police Ford Crown Victoria - Police Planes Cessna 206 Website http://www.chp.ca.gov http://www.chpcareers.com The
California Highway Patrol ( CHP ) is a law enforcement agency of the state of California . The primary mission of the CHP is to ensure safety and enforce
traffic laws on all California highways and county roads in unincorporated areas . The CHP promotes traffic safety by inspecting commercial vehicles , as well
as inspecting and certifying school buses , ambulances , and other specialized vehicles . The California State Legislature established the California Highway
Patrol as a branch of the Division of Motor Vehicles in the Department of Public Works , with legislation signed by Governor C. C. Young on August 14 , 1929 .
</item 6>

<item 7>
California Highway Patrol
Vehicles ( edit ) This section needs expansion . You can help by adding to it . ( September 2013 ) Origins ( edit ) When motor vehicles in California were first
seen as needing legislation , law enforcement agencies began to patrol using motorcycles , cars and trucks . Motorcycle officers in 1920 Fresno , started a
group to assist each other and promote road safety - the Joaquin Valley Traffic Officer 's Association - led by Harry Wilson who they elected as president .
They renamed themselves the California Association of Highway Patrolmen in 1921 , and became the California Highway Patrol in 1927 under the auspices of the
Department of Motor Vehicles . Motorcycles ( edit ) A motorcycle sergeant on the San Tomas Expressway near Benton Avenue in Santa Clara , California . Through
the public competitive bidding process , the Harley - Davidson motorcycle was selected as the primary enforcement motorcycle for the California Highway Patrol
in 2013 . These replacement enforcement motorcycles will replenish the Department 's aging motorcycle fleet . In a cost - saving move , the CHP previously
deferred the purchase of replacement motorcycles and has not purchased enforcement motorcycles since January 14 , 2011 .
</item 7>

<item 8>
California Highway Patrol
This is because of their ability to carry a larger `` basic load '' of normal issue equipment , equipment that may be only in a few cars , and / or not carried
by patrol cars at all , in a more urban setting . Aircraft ( edit ) CHP AS 350 B3 in flight . The CHP utilizes both helicopters and fixed - wing aircraft . In
2017 the fleet consisted of : 15 Airbus Helicopters ( Eurocopter ) AS350B3 Squirrels 11 Cessna T206H Turbo Stationairs 4 Gippsaero GA8 TC - 320 Airvans 2 Hughes
( MD500 ) 369Ds 1 Cessna 182S Skylane 1 Raytheon ( Beechcraft ) B300 Super King Air Rogue officers ( edit ) George Gwaltney ( edit ) A former officer from
Barstow , was convicted on May 10 , 1984 , in federal court , for the on ‐ duty rape and slaying of 23 - year - old Robin Bishop on January 11 , 1982 .
</item 8>

<item 9>
California Highway Patrol
Weapons ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced
material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) The previous department issued firearm for
CHP officers is the Smith & Wesson Model 4006 TSW in . 40 S&W . As of August 2017 , the department has begun replacing the 4006 TSW with the Smith & Wesson M&P
in . 40 S&W . Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12 ‐ gauge shotgun and a Sig Sauer M400 rifle in 5.56 mm . Less Lethal
options for officers include OC Pepper Spray and the PR - 24 Side Handle Baton . The ASP Expandable Straight Baton is authorized as an optional piece of
equipment for officers completing their probationary year . Additionally , officers are authorized to carry the TASER X2 Defender . In early 2009 , officers
have been allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols .
</item 9>

<item 10>
California Highway Patrol
ESPN March 6 2008 Jump up ^ Game - warden shortage is about to get worse San Francisco Chronicle September 23 2007 Archived February 15 , 2012 , at the Wayback
Machine . Jump up ^ Lots of ocean , but few game wardens ! Sacramento Bee August 23 2007 Archived August 21 , 2007 , at the Wayback Machine . Jump up ^ Oregon
State Police Fish and Wildlife Division Archived 2012 - 07 - 31 at the Wayback Machine . Jump up ^ Division of Alaska Wildlife Troopers Jump up ^
http://smith-wessonforum.com/showthread.php?t=525162 ^ Jump up to : Mattos , Rick ( 2008 ) . California Highway Patrol . Charleston , SC : Arcadia Pub . ISBN
9780738556208 . Retrieved 2013 - 09 - 01 . Jump up ^ Carpenter , Susan ( 25 June 2013 ) . `` BMW 's R1200RT - P : Big , bad and packing heat '' . Los Angeles
Times . Retrieved 1 September 2013 .
</item 10>
=== END ITEM id=f55cbe62c1d39b92 ===

=== ITEM id=d32bf78e9fec17e9 ===
Gold sentences and facts:
Sentence 0: The Battle of Lake Erie , sometimes called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval battles of the War of 1812 .
  Fact 0.0: The Battle of Lake Erie is sometimes called the Battle of Put-in-Bay.
  Fact 0.1: The Battle of Lake Erie was fought on 10 September 1813.
  Fact 0.2: The Battle of Lake Erie was fought in Lake Erie off the coast of Ohio.
  Fact 0.3: The Battle of Lake Erie was fought during the War of 1812.
  Fact 0.4: Nine vessels of the United States Navy defeated six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.5: Nine vessels of the United States Navy captured six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.6: The Battle of Lake Erie ensured American control of the lake for the rest of the War of 1812.
  Fact 0.7: The Battle of Lake Erie allowed the Americans to recover Detroit.
  Fact 0.8: The Battle of Lake Erie allowed the Americans to win the Battle of the Thames.
  Fact 0.9: Winning the Battle of the Thames broke the Indian confederation of Tecumseh.
  Fact 0.10: The Battle of Lake Erie was one of the biggest naval battles of the War of 1812.

Retrieved items:
<item 1>
Battle of Lake Erie
ISBN 0 - 8117 - 2876 - 5 . Roosevelt , Theodore ( 2004 ) . The Naval War of 1812 . New York : The Modern Library . ISBN 0 - 375 - 75419 - 9 . Skaggs , David ;
Atloff , Gerard ( 1997 ) . A Signal Victory : The Lake Erie Campaign , 1812 -- 13 . Naval Institute Press . ISBN 1 - 55750 - 892 - 5 . Symonds , Craig ( 2005 )
. Decision at Sea . Oxford University Press . ISBN 0 - 19 - 517145 - 4 . Zaslow , Morris ( ed ) . ( 1964 ) . The Defended Border : Upper Canada and the War of
1812 . Toronto : Macmillan of Canada . ISBN 0 - 7705 - 1242 - 9 . Further reading ( edit ) Axelrod , Alen ; Phillips , Charles . The MacMillan Dictionary of
Military Biography ( New York : MacMillan , 1998 . )
</item 1>

<item 2>
Battle of Lake Erie
USS United States vs HMS Macedonian Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie Battle of Fort Oswego Engagements on Lake Huron
Engagements on Lake Ontario First Battle of Sacket 's Harbor Gulf Coast Action of 13 December 1814 Battle of Lake Borgne Pacific Ocean Action off James Island
Action off Charles Island Nuku Hiva Campaign Battle of Valparaiso ( Capture of USS Essex ) See also : American Indian Wars , Creek War , Napoleonic Wars , and
Tecumseh 's War Portal definition textbooks quotes source texts media news stories Coordinates : 41 ° 39 ′ 44 '' N 82 ° 49 ′ 30 '' W ﻿ / ﻿ 41.66222 ° N 82.82500
° W ﻿ / 41.66222 ; - 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Battle of Lake Erie
James Massacre Missouri Battle of Credit Island Battle of the Sink Hole British North America Lower Canada Battle of the Chateauguay First Battle of Lacolle
Mills Second Battle of Lacolle Mills Upper Canada Battle of Beaver Dams Battle of Chippawa Battle of Cook 's Mills Battle of Crysler 's Farm Battle of Fort
George Battle of Frenchman 's Creek Battle of Longwoods Battle of Lundy 's Lane Battle of Malcolm 's Mills Battle of Queenston Heights Battle of Stoney Creek
Battle of the Thames Battle of York Capture of Fort Erie Raid on Elizabethtown Raid on Port Dover Raid on Gananoque Siege of Fort Erie Spanish Empire Spanish
Florida Battle of Pensacola Naval battles Atlantic Ocean Capture of HMS Boxer Capture of HMS Cyane Capture of HMS Epervier Capture of HMS Frolic Capture of HMS
Penguin Capture of HMS Dominica Capture of USS Argus Capture of USS Chesapeake Capture of USS President Chesapeake Bay Flotilla USS Constitution vs HMS Java
Sinking of HMS Avon Battle of Fayal Sinking of HMS Peacock Sinking of HMS Reindeer USS Constitution vs HMS Guerriere USS United States vs HMS Macedonian
Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie
</item 3>

<item 4>
Battle of Lake Erie
External links ( edit ) Wikimedia Commons has media related to Battle of Lake Erie . United States Navy portal Rhode Island portal Erie portal Ohio portal War
of 1812 portal History of Canada portal Wikiquote has quotations related to : Battle of Lake Erie `` Log of the Battle of Lake Erie '' by Sailing Master William
Taylor US Brig Niagara Barclay 's court martial records and correspondence Battle of Lake Erie Historical Sources Animated history of the Battle of Lake Erie ``
Erie , Lake , Battle of '' . The American Cyclopædia. 1879 . Conflicts of the War of 1812 Battles of the War of 1812 United States Washington , D.C. Burning of
Washington Georgia Battle of Fort Peter Louisiana Battle of New Orleans Siege of Fort St. Philip Maryland Battle of Baltimore Battle of Bladensburg Battle of
Caulk 's Field Battle of North Point Battle of St.
</item 4>

<item 5>
Battle of Lake Erie
Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS Guerriere Capture of HMS Frolic USS United States vs HMS
Macedonian USS Constitution vs HMS Java Sinking of HMS Peacock Rappahannock River Capture of USS Chesapeake Capture of HMS Dominica Capture of USS Argus Capture
of HMS Boxer Capture of USS Frolic Capture of HMS Epervier Sinking of HMS Reindeer Sinking of HMS Avon Fayal Capture of USS President Capture of HMS Cyane and
HMS Levant Capture of HMS Penguin Capture of East India Company ship Nautilus East Coast Chesapeake Bay Alexandria Baltimore Hampden Fort Peter Great Lakes /
Saint Lawrence River Lake Ontario 1st Sacket 's Harbor York Fort George 2nd Sacket 's Harbor Lake Erie Fort Oswego Lake Huron Lake Champlain West Indies / Gulf
Coast La Guaira 1st Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St.
</item 5>

<item 6>
Battle of Lake Erie
It was one of the biggest naval battles of the War of 1812 . Contents ( hide ) 1 Background 1.1 1812 1.2 1813 1.3 Blockades of Presque Isle and Amherstburg 2
Battle 3 Casualties 4 Aftermath 5 Reconstructions and memorials 6 Reasons for the American victory 7 Order of battle 8 Footnotes 9 Notes 10 References 11
Further reading 12 External links Background ( edit ) 1812 ( edit ) When the war broke out , the British immediately seized control of Lake Erie . They already
had a small force of warships there : the sloop - of - war Queen Charlotte and the brig General Hunter . The schooner Lady Prevost was under construction and
was put into service a few weeks after the outbreak of war . These vessels were controlled by the Provincial Marine , which was a military transport service and
not a naval service . Nevertheless , the Americans lacked any counter to the British armed vessels .
</item 6>

<item 7>
Battle of Lake Erie
Battle of Lake Erie - wikipedia Battle of Lake Erie Jump to : navigation , search Battle of Lake Erie Part of the War of 1812 Battle of Lake Erie by William
Henry Powell , painted 1865 , shows Oliver Hazard Perry transferring from Lawrence to Niagara Date 10 September 1813 Location Lake Erie , near Put - in - Bay ,
Ohio Result Decisive American victory Belligerents United States United Kingdom Commanders and leaders Oliver Hazard Perry Jesse Elliot Robert Heriot Barclay
Robert Finnis Strength 5 schooners 3 brigs 1 sloop 2 ships 1 brig 2 schooners 1 sloop Casualties and losses 27 killed 96 wounded 1 brig severely damaged 41
killed 93 wounded 306 captured entire squadron captured Detroit frontier Tippecanoe Fort Mackinac ( 1812 ) Brownstown Maguaga Fort Dearborn Detroit Fort
Harrison Fort Wayne Wild Cat Creek Mississinewa Frenchtown Africa Point Fort Meigs Fort Stephenson Lake Erie Thames Longwoods Prairie du Chien Campbell Island
Mackinac Island ( 1814 ) Lake Huron Malcolm 's Mills Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS
</item 7>

<item 8>
Battle of Lake Erie
Michaels Raid on Havre de Grace Massachusetts Battle of Hampden New York Battle of Big Sandy Creek Battle of Buffalo Battle of Ogdensburg Battle of Plattsburgh
Capture of Fort Niagara Raid on Black Rock Second Battle of Sacket 's Harbor Ohio Battle of Fort Stephenson Copus massacre Siege of Fort Meigs Virginia Battle
of Craney Island Raid on Alexandria Skirmish at Farnham Church Battle of Rappahannock River U.S. territories Alabama Battles of Fort Bowyer Illinois Battle of
Fort Dearborn Battle of Rock Island Rapids Siege of Prairie du Chien Indiana Battle of the Mississinewa Battle of Tippecanoe Battle of Wild Cat Creek Siege of
Fort Harrison Siege of Fort Wayne Michigan Battle of Brownstown Battle of Frenchtown Battle of Mackinac Island Battle of Maguaga Siege of Detroit Siege of Fort
Mackinac Mississippi Battle of Burnt Corn Battle of Callabee Creek Canoe Fight Battle of Holy Ground Battle of Horseshoe Bend Battle of Talladega Battle of
Tallushatchee Battles of Emuckfaw and Enotachopo Creek Fort Mims massacre Kimbell -- James Massacre Missouri Battle of Credit Island Battle of the Sink Hole
British North America Lower Canada Battle of
</item 8>

<item 9>
Battle of Lake Erie
Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St. Philip 2nd Fort Bowyer Pacific Ocean James Island Charles Island Nuku Hiva Downes
Expedition Porter Expedition Typee Valley Valparaiso ( Capture of USS Essex ) Seringapatam Mutiny Action of 9 May 1814 The Battle of Lake Erie , sometimes
called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United
States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn
allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval
battles of the War of 1812 .
</item 9>

<item 10>
Battle of Lake Erie
- 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Battle_of_Lake_Erie&oldid=799568331 '' Categories : Conflicts in 1813 Battles involving
Canada History of Ohio Naval battles of the War of 1812 Naval battles involving Canada Lake Erie 1813 in Ohio September 1813 events Hidden categories : CS1
maint : Extra text : authors list Pages using deprecated image syntax Articles needing additional references from September 2017 All articles needing additional
references Wikipedia articles incorporating a citation from The American Cyclopaedia Wikipedia articles incorporating a citation from The American Cyclopaedia
with a Wikisource reference Coordinates not on Wikidata Use dmy dates from May 2011 Talk Contents About Wikipedia Afrikaans Català Deutsch Español Français 한국어
Magyar Bahasa Melayu 日本 語 Polski Português ไทย 中文 Edit links This page was last edited on 8 September 2017 , at 14 : 22 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply .
</item 10>
=== END ITEM id=d32bf78e9fec17e9 ===

=== ITEM id=1ee868a26bca4a9f ===
Gold sentences and facts:
Sentence 0: John Ernest Crawford ( born March 26 , 1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national audience as a Mouseketeer .
  Fact 0.0: John Ernest Crawford was born March 26, 1946.
  Fact 0.1: John Ernest Crawford is an American character actor.
  Fact 0.2: John Ernest Crawford is an American singer.
  Fact 0.3: John Ernest Crawford is an American musician.
  Fact 0.4: At age 12, John Ernest Crawford rose to fame for playing Mark McCain.
  Fact 0.5: Mark McCain is the son of Lucas McCain.
  Fact 0.6: Lucas McCain was played by Chuck Connors.
  Fact 0.7: Crawford played Mark McCain in the popular ABC Western series The Rifleman.
  Fact 0.8: The Rifleman originally aired from 1958 to 1963.
  Fact 0.9: John Ernest Crawford first performed before a national audience as a Mouseketeer.

Retrieved items:
<item 1>
Johnny Crawford
His first important break as an actor followed with the title role in a Lux Video Theatre production of `` Little Boy Lost '' , a live NBC broadcast on March 15
, 1956 . He also appeared in the popular Western series The Lone Ranger , in 1956 , in one of the few color episodes of that series . Following that performance
, the young actor worked steadily with many seasoned actors and directors . Freelancing for 2 1 / 2 years , he accumulated almost 60 television credits ,
including featured roles in three episodes of NBC 's The Loretta Young Show and an appearance as Manuel in , `` I Am an American '' , an episode of the
syndicated crime drama Sheriff of Cochise . By the spring of 1958 , he had also performed 14 demanding roles in live teleplays for NBC 's Matinee Theatre ,
appeared on CBS 's sitcom , Mr. Adams and Eve , in the Wagon Train episode `` The Sally Potter Story '' ( in which Martin Milner also appeared ) and on the
syndicated series , Crossroads , Sheriff of Cochise , and Whirlybirds , and made three pilots of TV series .
</item 1>

<item 2>
Johnny Crawford
The third pilot , which was made as an episode of Dick Powell 's Zane Grey Theater , was picked up by ABC and the first season of The Rifleman began filming in
July 1958 . Crawford was nominated for an Emmy Award , at age 13 , for his role as Mark McCain , the son of Lucas McCain , played by Chuck Connors , in the Four
Star Television series The Rifleman , which originally aired from 1958 to 1963 . Throughout The Rifleman 's five seasons , a remarkable on - screen chemistry
existed between Connors and Crawford in the depiction of their father - son relationship . They were still close friends when Connors died on November 10 , 1992
, and Crawford gave a eulogy at Connors ' memorial service . During the late 1950s and early 1960s , Crawford had wide popularity with American teenagers and a
recording career that generated four Billboard Top 40 hits , including the single , `` Cindy 's Birthday '' , which peaked at number 8 , in 1962 .
</item 2>

<item 3>
Johnny Crawford
Johnny Crawford - wikipedia Johnny Crawford Jump to : navigation , search For the Canadian ice hockey player and coach , see Johnny Crawford ( ice hockey ) .
Johnny Crawford Johnny Crawford in The Rifleman ( 1961 ) John Ernest Crawford ( 1946 - 03 - 26 ) March 26 , 1946 ( age 72 ) Los Angeles , California , U.S.
Occupation Actor , singer , musician , band leader Years active 1954 -- 96 Spouse ( s ) Charlotte Crawford ( 1995 -- present ) ( separated ) Relatives Robert L.
Crawford , Jr. ( brother ) Website Johnny Crawford fansite Johnny Crawford and Chuck Connors in The Rifleman ( 1960 ) John Ernest Crawford ( born March 26 ,
1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by
Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national
audience as a Mouseketeer .
</item 3>

<item 4>
Johnny Crawford
He played a young deputy Billy Norris , in The Big Valley episode `` The Other Face Of Justice '' in 1969 . While enlisted in the United States Army for two
years , Crawford worked on training films as a production coordinator , assistant director , script supervisor , and occasional actor . His rank was sergeant at
the time of his honorable discharge in December 1967 . In 1968 , Crawford played a soldier wanted for murder in `` By the Numbers '' , an episode of the popular
TV series Hawaii Five - O . The Resurrection of Broncho Billy was a USC student film Crawford agreed to do as a favor to his close friend , producer John
Longenecker . It won the 1970 Academy Award for Best Live Action Short Subject . The Naked Ape was a partially animated 1973 feature film starring Crawford and
Victoria Principal , produced by Hugh Hefner . In 1976 , Johnny Crawford co-starred as Ben Shelby in the tenth episode of season three of Little House on the
Prairie , `` The Hunters '' . In 1985 , Crawford appeared as Noah Paisley , sheriff 's deputy , in an episode of Murder , She Wrote .
</item 4>

<item 5>
Rio Bravo (film)
In this film , Robert Mitchum played the Dean Martin role , Arthur Hunnicutt the Walter Brennan character , and James Caan the Ricky Nelson role . Hawks again
named the Nelson / Caan character after a state ( in this case , Mississippi ) and in a wry , humorous twist on the original film , Hawks made him inept with
firearms , but skilled with a knife . The second remake , Rio Lobo , was made in 1970 with a plot much further off the original mold , starting with the absence
of a lawman - turned - drunkard character . This began with a Confederate train robbery of a Union gold shipment during the American Civil War , then moved to a
postwar Texas county thoroughly controlled by a rich , arrogant rancher . The heroes , with the exception of an old man similar to Brennan and Hunnicutt 's
characters in the previous pictures ( Jack Elam here ) , were complete outsiders . Along with Wayne and Elam , this movie starred Mexican film star Jorge Rivero
( as Frenchie ) , Christopher Mitchum ( Robert Mitchum 's son ) , and Jennifer O'Neill .
</item 5>

<item 6>
Rio Bravo (film)
After a pair of sheer tights comes floating from a window to the street , a delighted Stumpy retrieves it , but Dude cautions him to mind his own business .
Cast ( edit ) John Wayne and Angie Dickinson in Rio Bravo John Wayne as John T. Chance Dean Martin as Dude Ricky Nelson as Colorado / Ryan Angie Dickinson as
Feathers Walter Brennan as Stumpy Ward Bond as Pat Wheeler John Russell as Nathan Burdette Pedro Gonzalez Gonzalez as Carlos Robante Estelita Rodriguez as
Consuela Robante Claude Akins as Joe Burdette Joseph Shimada as Burt Malcolm Atterbury and Harry Carey Jr. also receive a screen credit in the film 's opening ,
but their scenes were deleted from the final film . Production ( edit ) Ricky Nelson performing the song `` Get Along Home , Cindy '' in Rio Bravo Exteriors for
the film were shot at Old Tucson Studios , just outside Tucson . Filming took place in the summer of 1958 , and the movie 's credits gave 1958 as the year of
production , although the film was not released until 1959 .
</item 6>

<item 7>
Rio Bravo (film)
Because the film starred a crooner , Martin , and a teen idol , Nelson , Hawks included three songs in the soundtrack . Before the big showdown , in the jail
house , Martin sings `` My Rifle , My Pony , and Me '' ( which contained new lyrics to a Tiomkin tune that appeared in Red River ) accompanied by Nelson , after
which Nelson sings a brief version of `` Get Along Home , Cindy '' , accompanied by Martin and Brennan . Over the closing credits , Martin , backed by the
Nelson Riddle Orchestra , sings a specially composed song , `` Rio Bravo '' , written by Tiomkin with lyrics by Paul Francis Webster . Nelson later paid homage
to both the film and his character , Colorado , by including the song `` Restless Kid '' on his 1959 LP , Ricky Sings Again . Members of the Western Writers of
America chose `` My Rifle , My Pony , and Me '' as one of the Top 100 Western songs of all time . A brief clip from Rio Bravo was among the archive footage
later incorporated into the opening sequence of Wayne 's last film , The Shootist , to illustrate the backstory of Wayne 's character .
</item 7>

<item 8>
Rio Bravo (film)
raˈtʃon ) , Spanish for `` big drunk '' ) , enters a saloon but can not afford a drink . Joe Burdette ( Claude Akins ) , brother of rancher Nathan Burdette ,
watches as a barfly ( Myron Healey ) , tosses a silver dollar into a spittoon . Presidio County , Texas , Sheriff John T. Chance ( John Wayne ) appears and
kicks the spittoon away , looking at Dude with disgust . Dude is shamed by his plight and takes out his anger on Chance , knocking him out with an ax handle .
Joe begins punching Dude , then shoots and kills an unarmed bystander who tries to intervene . Joe heads to his brother 's saloon , where a bloody Chance comes
to arrest him for the murder of the bystander . Another patron draws his gun on Chance , but Dude shoots the gun out of the man 's hand . Joe is locked up in
the local jail . Chance is willing to deputize Dude , provided he can stay sober . Both remember how good with a gun Dude used to be .
</item 8>

<item 9>
Johnny Crawford
Jump up ^ `` Johnny Crawford '' . Scott Stander . Retrieved January 25 , 2014 . Jump up ^ `` Angry Woman -- Victoria Jackson on `` Tonight Show '' b / w `` .
Youtube . Retrieved January 25 , 2014 . Biography portal California portal Los Angeles portal Film portal Television portal Music portal United States Army
portal External links ( edit ) Johnny Crawford on The Rifleman Broncho Billy short film Johnny Crawford 's MySpace site Johnny Crawford on IMDb VIAF : 63364407
LCCN : no91003539 ISNI : 0000 0001 1526 4402 GND : 142142425 SUDOC : 079205798 BNF : cb11219552n ( data ) MusicBrainz : bf40b9eb - 068a - 4bdd - be46 -
ec0b5bd7a177 BNE : XX4832002 SNAC : w6sk5xbc Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Rio Bravo (film)
Both remember how good with a gun Dude used to be . Chance 's friend Pat Wheeler ( Ward Bond ) and his wagon train of supplies stop in town , with a young
gunslinger , Colorado Ryan ( Ricky Nelson ) , riding guard . Inside the jail , Stumpy ( Walter Brennan ) , Chance 's game - legged deputy , keeps watch over the
jail and Joe , who knows that Stumpy holds an old grudge against Joe 's wealthy and powerful brother . Joe warns his jailers that Nathan Burdette will not like
how his brother is being treated . A mysterious woman nicknamed Feathers ( Angie Dickinson ) is in the saloon , playing poker . In the meantime , Dude and
Chance patrol the town . Hotel owner Carlos ( Pedro Gonzalez - Gonzalez ) stops the sheriff , saying Wheeler has been talking too much about Chance needing help
. In the saloon , Chance implores Wheeler to stop , as it will draw attention from the wrong people . Wheeler suggests that Colorado could be of assistance ,
but Colorado politely declines , saying he wants to `` mind my own business '' . Colorado also promises not to start any trouble without telling the sheriff
first .
</item 10>
=== END ITEM id=1ee868a26bca4a9f ===
