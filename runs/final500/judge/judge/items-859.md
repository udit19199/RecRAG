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

=== ITEM id=2ee024191294fb18 ===
Gold sentences and facts:
Sentence 0: The first installment , Divergent ( 2014 ) , grossed over $288 million worldwide , while the second installment , The Divergent Series : Insurgent ( 2015 ) , grossed over $297 million worldwide . Insurgent was also the first Divergent film to be released in IMAX 3D . The third installment , The Divergent Series : Allegiant ( 2016 ) , grossed $179 million . Thus , the first three films of the series have grossed over $765 million worldwide . A fourth film , The Divergent Series : Ascendant was never made , due to Allegiant 's poor showing at the box office .
  Fact 0.0: Divergent (2014) is the first installment of the Divergent series.
  Fact 0.1: Divergent grossed over $288 million worldwide.
  Fact 0.2: The Divergent Series: Insurgent (2015) is the second installment of the Divergent series.
  Fact 0.3: The Divergent Series: Insurgent grossed over $297 million worldwide.
  Fact 0.4: The Divergent Series: Insurgent was the first Divergent film released in IMAX 3D.
  Fact 0.5: The Divergent Series: Allegiant (2016) is the third installment of the Divergent series.
  Fact 0.6: The Divergent Series: Allegiant grossed $179 million.
  Fact 0.7: The first three films of the Divergent series have grossed over $765 million worldwide in total.
  Fact 0.8: The Divergent Series: Ascendant is a fourth film that was never made.
  Fact 0.9: The Divergent Series: Ascendant was never made due to The Divergent Series: Allegiant's poor showing at the box office.

Retrieved items:
<item 1>
The Divergent Series
Veronica Roth and Neil Burger have acted as executive producers on the first two films . Summit Entertainment and Lionsgate collaborated on all four films . On
April 11 , 2014 , the studio decided to split the final book , Allegiant , into two parts , much like Harry Potter and the Deathly Hallows -- Parts 1 and 2 ,
The Twilight Saga : Breaking Dawn -- Parts 1 and 2 , and The Hunger Games : Mockingjay -- Parts 1 and 2 . Directors ( edit ) Neil Burger directed the first film
, Divergent , and was originally going to direct the sequel , The Divergent Series : Insurgent , but in December 2013 it was announced that Burger would not
return , due to him still working on post-production work for the first film . Speaking on the subject the director stated , `` There was no breathing room .
They were like , ' We 're holding to a March 2015 release date , and we have to start no later than May 2014 . ' It was a recipe for failure , at least with me
on board . I would n't be able to do either thing right .
</item 1>

<item 2>
The Divergent Series
by Veronica Roth Composer Junkie XL Joseph Trapanese N / A Cinematographer Alwin H. Küchler Florian Ballhaus Editor ( s ) Richard Francis - Bruce Nancy
Richardson Nancy Richardson Stuart Levy Stuart Levy N / A Production companies Red Wagon Entertainment Summit Entertainment Red Wagon Entertainment Summit
Entertainment Mandeville Films Red Wagon Entertainment Mandeville Films N / A Distributor Lionsgate Summit Entertainment N / A Running time 139 minutes 119
minutes 121 minutes TBA Release date March 21 , 2014 March 20 , 2015 March 18 , 2016 TBA Reception ( edit ) Further information on the reception of each film :
Divergent ( film ) § Reception , The Divergent Series : Insurgent § Reception , and The Divergent Series : Allegiant § Reception Though the first two films were
commercially successful , the series as a whole has not been well received by critics . Each successive film release has been rated lower than the previous
release on review aggregator websites . Negative reviews of the third novel Allegiant and the studios ' decision to split it into two films have been blamed for
the third film 's lackluster performance at the box office .
</item 2>

<item 3>
The Divergent Series
They have been produced by Lucy Fisher , Pouya Shabazian , and Douglas Wick . The series star Shailene Woodley and Theo James as lead characters Beatrice Prior
( Tris ) and Tobias Eaton ( Four ) , respectively . The supporting cast includes Ansel Elgort , Zoë Kravitz , and Miles Teller . Kate Winslet played the main
antagonist in the first two films . The first film in the series was directed by Neil Burger , while the second and third films were directed by Robert
Schwentke . The Veronica Roth novels consist primarily of the trilogy ( ' Divergent ' , ' Insurgent ' , ' Allegiant ' ) . Development began in 2011 following
Summit 's acquisition of the film rights to the Divergent novel in partnership with production company Red Wagon Entertainment . The studios announced
production on the sequel following the first film 's strong performance in Thursday late - night screenings , where it grossed $4.9 million . They acquired film
rights to the Allegiant novel in December 2013 , deciding in April 2014 to split the third novel into a two - part film adaptation .
</item 3>

<item 4>
The Divergent Series
The first installment , Divergent ( 2014 ) , grossed over $288 million worldwide , while the second installment , The Divergent Series : Insurgent ( 2015 ) ,
grossed over $297 million worldwide . Insurgent was also the first Divergent film to be released in IMAX 3D . The third installment , The Divergent Series :
Allegiant ( 2016 ) , grossed $179 million . Thus , the first three films of the series have grossed over $765 million worldwide . A fourth film , The Divergent
Series : Ascendant was never made , due to Allegiant 's poor showing at the box office .
</item 4>

<item 5>
The Divergent Series
The Divergent Series : Allegiant ( 2016 ) ( edit ) Main article : The Divergent Series : Allegiant On December 16 , 2013 , Summit Entertainment announced that
the film adaptation of Allegiant would be released on March 18 , 2016 . On April 11 , 2014 , Lionsgate announced that the film adaptation would be split into
two films with the first part titled , The Divergent Series : Allegiant -- Part 1 . The film was then re-titled as The Divergent Series : Allegiant . Tris must
escape with Four and go beyond the wall enclosing Chicago . For the first time ever , they will leave the only city and family they have ever known in order to
find a peaceful solution for their embroiled city . Once outside , old discoveries are quickly rendered meaningless with the revelation of shocking new truths .
Tris must quickly decide who she can trust as a ruthless battle ignites beyond the walls of Chicago . In order to survive , Tris will be forced to make
impossible choices about courage , allegiance , sacrifice and love .
</item 5>

<item 6>
The Divergent Series
For the mathematical use , see Divergent series . The Divergent Series Produced by Lucy Fisher Douglas Wick Pouya Shabazian Based on The Divergent trilogy by
Veronica Roth Starring Shailene Woodley Theo James Zoë Kravitz Miles Teller Ansel Elgort Production company Red Wagon Entertainment Summit Entertainment
Distributed by Summit Entertainment Lionsgate Release date 1 : March 21 , 2014 ( 2014 - 03 - 21 ) 2 : 000000002015 - 03 - 20 - 0000 March 20 , 2015 3 :
000000002016 - 03 - 18 - 0000 March 18 , 2016 4 : Cancelled Running time 379 minutes ( 3 films ) Country United States Language English Budget Total ( 3 films )
: $305 million Box office Total ( 3 films ) : $765,409,015 The Divergent Series is a feature film trilogy based on the Divergent novels by the American author
Veronica Roth . Distributed by Summit Entertainment and Lionsgate Films , the series consists of three science fiction films set in a dystopian society .
</item 6>

<item 7>
The Divergent Series
I 'm thrilled ! '' The producers said about his casting : `` We took our time to find the right actor to fill the role of Four , and Theo is definitely the
perfect fit . Veronica has crafted a truly iconic character in Four and we can not wait to begin production and bring him and this story to life for millions of
fans around the world . '' Production ( edit ) Filming for the franchise began on April 16 , 2013 . Film Producers Director Screenwriters Novel by Veronica Roth
Divergent Fisher , Lucy Lucy Fisher Douglas Wick Pouya Shabazian Burger , Neil Neil Burger Evan Daugherty Vanessa Taylor Divergent The Divergent Series :
Insurgent Schwentke , Robert Robert Schwentke Akiva Goldsman Brian Duffield Insurgent The Divergent Series : Allegiant Noah Oppenheim Adam Cooper Bill Collage
Allegiant The Divergent Series : Ascendant Lee Toland Krieger Lucy Fisher , Douglas Wick , and Pouya Shabazian have been producers for all four films in the
series . Veronica Roth and Neil Burger have acted as executive producers on the first two films .
</item 7>

<item 8>
The Divergent Series
Cancelled film ( edit ) A Third sequel to the film was initially planned , with a March 2017 release , based on the latter half of the Allegiant book . The
timing was changed to a June 2017 release , with Lee Toland Krieger then attached as the movie 's director . Ascendant ( TV Series ) ( edit ) In July 2016 ,
after the poor performance of the third film at the box office , the producers considered creating the fourth as a television film , which would add new
characters to the story who could continue in a spinoff television series , moving beyond the books . In September 2016 , Shailene Woodley stated on Today that
the film versus television decision was not finalized , and that it was `` a limbo waiting game '' . In the same interview Woodley spoke despairingly of the
chances of her returning to the project in a television format , although noting she would be open to returning to it as a theatrical film . In February 2017 ,
with the producers having confirmed that it would be a television project , it was announced that Woodley had backed out of her starring role .
</item 8>

<item 9>
The Divergent Series
Neil Burger was announced as the director on August 23 , 2012 . Evan Daugherty , who co-wrote the screenplay with Vanessa Taylor , said , `` I get hung up on
the toughness of the movie but of equal importance is the love story between Tris and Four . It 's inherently and inextricably linked to Tris ' character
journey . There will be plenty of sexual tension and chemistry , but it 's important that all of that stuff does n't just feel like it 's thrown in , but that
it all helps Tris grow as a character . '' Daugherty further added , `` It 's tricky because the book is a very packed read with a lot of big ideas . So ,
distilling that into a cool , faithful two - hour movie is challenging . Not only do you have to establish five factions , but you have to acknowledge that
there 's a sixth entity , which is the divergent , and you also have the factionless . So there 's a world that really has to be built out for the big screen
... the movie is going to do it a little more efficiently . ''
</item 9>

<item 10>
The Divergent Series
On December 16 , 2013 , it was announced that Neil Burger , director of Divergent , would not return to direct Insurgent as he was still working on the first
film . On February 13 , 2014 , it was announced that Robert Schwentke was offered the director position for the film and that Akiva Goldsman had been hired to
re-write Duffield 's script . Also in December 2013 , Summit Entertainment announced that the film adaptation of Allegiant , the third and final novel in the
Divergent trilogy , would be released in March 2016 , serving as the finale of the series , which at the time was planned as a trilogy , but on April 11 , 2014
, the studio decided to split the novel into a two - part film , much the like the Harry Potter , Twilight , and Hunger Games franchises did with the finales of
their series .
</item 10>
=== END ITEM id=2ee024191294fb18 ===

=== ITEM id=b72b700b4fa3334b ===
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
League of Nations
Race to the Sea First Battle of Ypres 1915 Second Battle of the Masurian Lakes Second Battle of Ypres Battle of Gallipoli Second Battle of Artois Battles of the
Isonzo Gorlice -- Tarnów Offensive Great Retreat Second Battle of Champagne Kosovo Offensive Siege of Kut Battle of Loos 1916 Erzurum Offensive Battle of Verdun
Lake Naroch Offensive Battle of Asiago Battle of Jutland Battle of the Somme first day Brusilov Offensive Baranovichi Offensive Battle of Romani Monastir
Offensive Battle of Transylvania 1917 Capture of Baghdad First Battle of Gaza Zimmermann Telegram Second Battle of Arras Second Battle of the Aisne Kerensky
Offensive Third Battle of Ypres ( Passchendaele ) Battle of Mărășești Battle of Caporetto Southern Palestine Offensive Battle of Cambrai Armistice of Erzincan
1918 Operation Faustschlag Treaty of Brest - Litovsk Spring Offensive Second Battle of the Marne Battle of Baku Hundred Days Offensive Vardar Offensive Battle
of Megiddo Third Transjordan attack Meuse - Argonne Offensive Battle of Vittorio Veneto Battle of Aleppo
</item 1>

<item 2>
Bulgaria during World War I
In addition , Germany and Austria - Hungary guaranteed the Bulgarian government a war loan of 200,000,000 francs and in case the war lasted longer than four
months , they guaranteed an additional supplementary loan . The third documented was concluded at the German Eastern military headquarters in Pless by the Chief
of the German General Staff Erich von Falkenhayn , the Chief of the Austro - Hungarian General Staff Count Franz Conrad von Hötzendorf and the delegate of the
Bulgarian government , Colonel Peter Ganchev . It was a military convention detailing the plan for the final defeat and conquest of Serbia . Germany and Austria
- Hungary were obliged to act against Serbia within thirty days of the signing of the convention , while Bulgaria had to do the same within 35 days of that date
. Germany and Austria - Hungary were to field at least six infantry divisions for the attack , and Bulgaria at least four infantry divisions according to their
established tables and organization . All these forces were to be placed under the command of Generalfeldmarschall August von Mackensen , whose task was `` to
fight the Serbian Army wherever he finds it and to open and insure as soon as possible a land connection between Hungary and Bulgaria '' .
</item 2>

<item 3>
Bulgaria during World War I
Strategic geographic location and a strong military establishment made the country a desired ally for both warring coalitions , but Bulgaria 's regional
aspirations were difficult to satisfy because they included territorial claims against four Balkan countries . As the war progressed , the Central Powers of
Austria - Hungary and the German Empire found themselves in a better position to fulfil Bulgarian demands and persuaded the country to join their cause in
September 1915 . Though the smallest member of the Central Powers in area and in population , Bulgaria made vital contributions to their common war effort . Its
entry to the war heralded the defeat of Serbia , thwarted the foreign - policy goals of Romania , and ensured the continuation of the Ottoman war effort by
providing a geographical conduit for material assistance from Germany to Istanbul . Though the Balkan theatre of the war saw successful campaigns of rapid
movement in 1915 and 1916 , the conflict degraded into a state of attritional trench warfare on both the Northern and the Southern Bulgarian Fronts after most
Bulgarian territorial aspirations had been satisfied . This period of the war substantially weakened the Bulgarian economy , created various supply problems and
reduced the health and morale of Bulgarian troops on the front lines .
</item 3>

<item 4>
Bulgaria during World War I
In reality , it hardly differed from the offer the Allies presented in May . Once again the promises lacked a clear guarantee that Serbia would cede the desired
lands and there was not even a mention of Southern Dobrudja . In the eyes of the Bulgarians , this was a manifestation of Entente helplessness in the face of
the conflicting ambitions of its smaller Balkan allies . The diplomatic positions of the Central Powers in Sofia were strengthened immensely , forcing the
Bulgarian tsar and prime minister to assume a course towards a final alignment of the country to the side of the Central Powers . In August , a Bulgarian
military mission led by Colonel Petar Ganchev , a former military attaché in Berlin , was dispatched to Germany to work out the details for a military
convention . Almost at the same time , the Minister of War Lieutenant General Ivan Fichev resigned and was replaced as minister by the pro-German Major General
Nikola Zhekov . Radoslavov also entered into talks with the Ottoman Empire , trying to gain concessions in exchange for Bulgarian benevolent neutrality .
</item 4>

<item 5>
League of Nations
Pacific and South Atlantic Oceans Papeete Madras Penang Cocos Coronel Falkland Islands Más a Tierra Principal participants ( people ) Entente powers Belgium
Brazil China France French Empire Greece Italy Japan Montenegro Portuguese Empire Romania Russia Russian Empire Russian Republic Serbia United Kingdom British
Empire United States Central Powers Germany Austria - Hungary Ottoman Empire Bulgaria Timeline Pre-War conflicts Scramble for Africa ( 1880 -- 1914 ) Russo -
Japanese War ( 1905 ) First Moroccan ( Tangier ) Crisis ( 1905 -- 06 ) Agadir Crisis ( 1911 ) Italo - Turkish War ( 1911 -- 12 ) French conquest of Morocco (
1911 -- 12 ) First Balkan War ( 1912 -- 13 ) Second Balkan War ( 1913 ) Prelude Origins Sarajevo assassination Anti-Serb riots in Sarajevo July Crisis Autumn
1914 Battle of the Frontiers Battle of Cer First Battle of the Marne Siege of Tsingtao Battle of Tannenberg Battle of Galicia Battle of the Masurian Lakes
Battle of Kolubara Battle of Sarikamish Race to the Sea First Battle of Ypres 1915 Second Battle of the Masurian Lakes
</item 5>

<item 6>
Bulgaria during World War I
By the middle of December , the entire Kingdom of Serbia was occupied by the armies of the Central Powers and the Allies were pushed back to Salonika by the
Bulgarians . Within two months of its entry into the war Bulgaria , achieved its main war goal : the conquest of Vardar Macedonia . For the duration of military
operations against Serbia and the Entente in 1915 , the Bulgarian Army committed a total of around 424,375 men , while its casualties were held to around 37,000
men . By the end of 1915 , the Central Powers had established firm and unbroken control over a vast territory that stretched from the North Sea to Mesopotamia .
They also drew great political and military dividends from the defeat and occupation of Serbia . Bulgaria conquered almost all the territory it desired ,
Germany gained unrestricted access to the natural resources of Ottoman Asia , the Ottomans received much needed German matériel assistance and Austria - Hungary
secured its southern flank and could completely focus its attention on the Russian and Italian fronts .
</item 6>

<item 7>
Bulgaria during World War I
The Germans and Austrians then undertook new counterattacks to restore their positions . Finally , in May 1915 , Italy entered the war on the side of the
Entente . Under these circumstances , the military and political value of the neutral Balkan countries increased significantly . The military successes of each
warring side were often a major asset in their diplomatic courtship of Bulgaria . Thus when Przemyśl fell and the Anglo - French landed in the Dardanelles ,
Radoslavov expressed greater interest in negotiations with the Entente . The leading role of Britain in the Gallipoli Campaign of 1915 made it a natural driving
force behind the revival of Entente attempts to acquire Bulgaria as an ally . The British realized that the key to winning Bulgaria was in Vardar Macedonia ,
and they suggested to Sazonov that Serbia should be prepared to cede the uncontested zone in exchange for Austrian territory . The Russian foreign minister
decided to stand behind this proposition , even though he found it rather vague , as long as it could turn Bulgaria against the Ottoman Empire .
</item 7>

<item 8>
Bulgaria during World War I
Allied representatives met with the leaders of the Bulgarian opposition parties and also provided generous financial support for opposition newspapers ; they
even attempted to bribe high - ranking government officials . Germany and Austria - Hungary were not willing to remain on the sidelines and dispatched to
Bulgaria Duke John Albert of Mecklenburg , the former ambassador to the Ottoman Empire Hans Freiherr von Wangenheim and Prince Hohenlohe , who openly declared
that after the defeat of Serbia , Bulgaria would assume hegemony of the Balkans . What seized Bulgarian interest the most was indeed the balance of military
power . The situation on the major European fronts was at that time developing markedly in favor of the Central Powers , and while the Allied operation in
Gallipoli turned into a costly stalemate , the Russians were being driven out of Galicia and Poland . Under these circumstances . the Central Powers were hoping
to secure Bulgaria at last . Still , it took Entente diplomacy more than a month to give an answer to Radoslavov 's questions and the reply proved far from
satisfactory . In reality , it hardly differed from the offer the Allies presented in May .
</item 8>

<item 9>
Bulgaria during World War I
The powers of the commander - in - chief were not regulated by law and even from the beginning , this caused some friction with the government . In his new role
, General Zhekov exercised direct control over all forces except those that remained in the interior of the country , which were placed under the command of the
new Minister of War Major General Kalin Naydenov . At the same time , Major General Konstantin Zhostov succeeded Lieutenant General Kliment Boyadzhiev , who was
appointed commander of the 1st Army as Chief of the Bulgarian General Staff . The military convention between Bulgaria and the Central Powers laid down the
general plan for its campaign against the Kingdom of Serbia . It severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army , which
was designated part of a combined German , Bulgarian and Austro - Hungarian force commanded by Field Marshal August von Mackensen . He had recently led the
German and Austro - Hungarian armies in the highly successful and victorious Gorlice -- Tarnów Offensive of the Central Powers against the Russian army on the
Eastern Front .
</item 9>

<item 10>
Bulgaria during World War I
In November , while the decisive Serbian defeat unfolded , the French attempted to exert pressure on the Bulgarian 2nd Army , but were soon forced to halt their
attempts to drive north . The forces of General Maurice Sarrail that consisted of three French and one British division dug in along an 80 - kilometer front
from the river Cherna to Lake Doiran . With the fall of Pristina , Generail Sarrail realized that the Allies could no longer help the Serbians and decided to
begin pulling back his forces to Salonika . The Bulgarian High Command shifted its focus to the Allies in Macedonia and decided that the time was right to go on
the offensive . Several days were lost , however , in scouting , and it was only on 3 December that the 2nd army commenced a general advance . Nonetheless , the
French were able to retreat in good order towards Salonika . They were soon followed by the British , who were defeated at the Battle of Kosturino . On 11
December , the Bulgarian divisions reached the Greek border , where they were ordered to halt and warned repeatedly not to cross .
</item 10>
=== END ITEM id=b72b700b4fa3334b ===

=== ITEM id=932aa3f46c9223d1 ===
Gold sentences and facts:
Sentence 0: State and Territory Capitals of Australia Jurisdiction Capital City Population State / Territory Population Date of statehood Capital since Image Australia and Australian Capital Territory Canberra 435,019 403,468 Must remain a territory as per s. 125 of the Australian Constitution 1913 View of Civic from Mount Ainslie , with Telstra Tower in the background New South Wales Sydney 5,029,768 7,759,274 1788 1788 Skyline of the Sydney central business district Victoria Melbourne 4,725,316 6,179,249 1851 1851 The Melbourne skyline on the Yarra River Queensland Brisbane 2,360,241 4,848,877 1859 1860 Brisbane 's CBD , with the Story Bridge in the foreground Western Australia Perth 2,022,044 2,558,951 1829 1829 View of Perth CBD skyline South Australia Adelaide 1,324,279 1,713,054 1836 1836 View of Adelaide Skyline . Tasmania Hobart 224,462 517,588 1825 1826 View of the Hobart downtown district and Mount Wellington from Constitution Dock Northern Territory Darwin 145,916 245,740 Has not attained Statehood 1911 View of Darwin CBD
  Fact 0.0: Canberra is the capital city of the Australian Capital Territory.
  Fact 0.1: Canberra has a population of 435,019.
  Fact 0.2: The Australian Capital Territory has a population of 403,468.
  Fact 0.3: The Australian Capital Territory must remain a territory as per s. 125 of the Australian Constitution.
  Fact 0.4: Canberra has been the capital since 1913.
  Fact 0.5: There is an image of a view of Civic from Mount Ainslie, with Telstra Tower in the background, for Canberra.
  Fact 0.6: Sydney is the capital city of New South Wales.
  Fact 0.7: Sydney has a population of 5,029,768.
  Fact 0.8: New South Wales has a population of 7,759,274.
  Fact 0.9: New South Wales attained statehood in 1788.
  Fact 0.10: Sydney has been the capital since 1788.
  Fact 0.11: There is an image of the Sydney central business district skyline for Sydney.
  Fact 0.12: Melbourne is the capital city of Victoria.
  Fact 0.13: Melbourne has a population of 4,725,316.
  Fact 0.14: Victoria has a population of 6,179,249.
  Fact 0.15: Victoria attained statehood in 1851.
  Fact 0.16: Melbourne has been the capital since 1851.
  Fact 0.17: There is an image of the Melbourne skyline on the Yarra River for Melbourne.
  Fact 0.18: Brisbane is the capital city of Queensland.
  Fact 0.19: Brisbane has a population of 2,360,241.
  Fact 0.20: Queensland has a population of 4,848,877.
  Fact 0.21: Queensland attained statehood in 1859.
  Fact 0.22: Brisbane has been the capital since 1860.
  Fact 0.23: There is an image of Brisbane's CBD, with the Story Bridge in the foreground, for Brisbane.
  Fact 0.24: Perth is the capital city of Western Australia.
  Fact 0.25: Perth has a population of 2,022,044.
  Fact 0.26: Western Australia has a population of 2,558,951.
  Fact 0.27: Western Australia attained statehood in 1829.
  Fact 0.28: Perth has been the capital since 1829.
  Fact 0.29: There is an image of the Perth CBD skyline for Perth.
  Fact 0.30: Adelaide is the capital city of South Australia.
  Fact 0.31: Adelaide has a population of 1,324,279.
  Fact 0.32: South Australia has a population of 1,713,054.
  Fact 0.33: South Australia attained statehood in 1836.
  Fact 0.34: Adelaide has been the capital since 1836.
  Fact 0.35: There is an image of the Adelaide skyline for Adelaide.
  Fact 0.36: Hobart is the capital city of Tasmania.
  Fact 0.37: Hobart has a population of 224,462.
  Fact 0.38: Tasmania has a population of 517,588.
  Fact 0.39: Tasmania attained statehood in 1825.
  Fact 0.40: Hobart has been the capital since 1826.
  Fact 0.41: There is an image of the Hobart downtown district and Mount Wellington from Constitution Dock for Hobart.
  Fact 0.42: Darwin is the capital city of the Northern Territory.
  Fact 0.43: Darwin has a population of 145,916.
  Fact 0.44: The Northern Territory has a population of 245,740.
  Fact 0.45: The Northern Territory has not attained statehood.
  Fact 0.46: Darwin has been the capital since 1911.
  Fact 0.47: There is an image of the Darwin CBD for Darwin.

Retrieved items:
<item 1>
List of Australian capital cities
List of Australian capital cities - wikipedia List of Australian capital cities Jump to : navigation , search There are eight capital cities in Australia , each
of which function as the seat of government for the state or territory in which they are located . Melbourne was the initial capital following the 1901
Federation of Australia . In 1913 , the seat of national government was moved to the newly created city of Canberra , which continues to serve as the national
capital . Each capital city hosts the judicial , administrative and legislative functions for their jurisdiction . In the case of the states and internal
territories , their capital cities also happen to be the most populous city in their respective jurisdiction . The Australian external territory of Norfolk
Island has its official capital at Kingston , although this acts merely as the administrative centre of government ; its de facto capital is Burnt Pine .
</item 1>

<item 2>
Prime Minister of Australia
Caretaker government Court of Disputed Returns Locations Parliament House , Melbourne Old Parliament House Parliament House , Canberra Politics of Australia
Commonwealth Monarch Governor - General Prime Minister Cabinet ( Shadow Cabinet ) Executive Council Ministry Foreign relations Parliament House of
Representatives Senate Opposition Leader High Court Lower courts Constitution Statute of Westminster Australia Act Federal elections pre-1969 1969 1972 1977
1980 1984 1990 1993 1998 2001 2007 2013 2016 next by - elections State / Territory governments Governors and Administrators Premiers and Chief Ministers
Parliaments and Assemblies NSW Vic Qld WA SA Tas ACT NT State / Territory elections Vic 2014 NSW 2015 ACT 2016 NT 2016 WA 2017 Qld 2017 Tas 2018 SA 2018 Vic
2018 NSW 2019 ACT 2020 NT 2020 WA 2021 Qld 2020 Tas 2022 SA 2022 Timeline of Elections Amalgamated Timeline Local government NSW Vic Qld WA SA Tas NT Political
parties Coalition ( Liberal , National , Liberal National , Country Liberal ) Conservatives Greens Hinch Katter Labor Liberal Democrat One Nation Xenophon Other
</item 2>

<item 3>
Foreign relations of Australia
Public Service Women Government leaders Government ministers Economy Agriculture Dollar ( currency ) Energy Gross state product Home ownership Public housing
Household income Manufacturing Media Mining Poverty Reserve Bank Stock Exchange Superannuation Taxation Telecommunications Tourism Transportation Welfare system
Society Crime Demographics Education Ethnic groups Healthcare Homelessness Immigration Languages Religion Women Culture Architecture Art Arts Australian English
Cinema Cuisine Dance Literature Music Public holidays Radio Sport Symbols Television Theatre Outline Index Book Portal Politics of Australia Commonwealth
Monarch Governor - General Prime Minister Cabinet ( Shadow Cabinet ) Executive Council Ministry Foreign relations Parliament House of Representatives Senate
Opposition Leader High Court Lower courts Constitution Statute of Westminster Australia Act Federal elections pre-1969 1969 1972 1977 1980 1984 1993 1998 2007
2013 2016 next by - elections State / Territory governments Governors and Administrators Premiers and Chief Ministers Parliaments and Assemblies NSW Vic Qld WA
SA Tas ACT NT State / Territory elections Vic 2014 SA 2014 Tas 2014 NSW 2015 ACT 2016 NT 2016 WA 2017 Qld 2017 Vic 2018 SA 2018 Tas 2018 NSW 2019 ACT 2020 NT
2020 WA 2021
</item 3>

<item 4>
List of Australian capital cities
125 of the Australian Constitution 1913 View of Civic from Mount Ainslie , with Telstra Tower in the background New South Wales Sydney 5,029,768 7,759,274 1788
1788 Skyline of the Sydney central business district Victoria Melbourne 4,725,316 6,179,249 1851 1851 The Melbourne skyline on the Yarra River Queensland
Brisbane 2,360,241 4,848,877 1859 1860 Brisbane 's CBD , with the Story Bridge in the foreground Western Australia Perth 2,022,044 2,558,951 1829 1829 View of
Perth CBD skyline South Australia Adelaide 1,324,279 1,713,054 1836 1836 View of Adelaide Skyline . Tasmania Hobart 224,462 517,588 1825 1826 View of the Hobart
downtown district and Mount Wellington from Constitution Dock Northern Territory Darwin 145,916 245,740 Has not attained Statehood 1911 View of Darwin CBD Notes
( edit ) Jump up ^ Although the ACT has its own government , it remains a territory ultimately under Commonwealth rule .
</item 4>

<item 5>
List of Australian capital cities
gov.au/AUSSTATS/abs@.nsf/DetailsPage/3218.02016?OpenDocument External links ( edit ) Capital cities of Australia National and ACT Canberra NSW Sydney NT Darwin
QLD Brisbane SA Adelaide TAS Hobart VIC Melbourne WA Perth List of cities in Australia Australia articles History Timeline Bibliography Prehistory Archaeology
European Exploration Territorial evolution 1788 -- 1850 1851 -- 1900 1901 -- 1945 Since 1945 By topic Constitutional Diplomatic Economic Federation Immigration
Indigenous Military Monarchy Railway Geography Climate Continent Deserts Environment issues Earthquakes Flora Fauna Forests Geology Islands Mountains Regions
Protected areas Rivers Subdivisions States and Territories Capitals Cities Politics Constitution Courts Elections Donations Electoral system Parties Federal
budget Foreign relations Government Governor - General Human rights Intersex LGBT Transgender Intelligence and security Law Constitution Judiciary Enforcement
Military Monarchy National debt Parliament Public Service Women Government leaders Government ministers Economy Agriculture Dollar ( currency ) Energy Gross
state product Home ownership Public housing Household income Manufacturing Media Mining Poverty Reserve Bank Stock Exchange Superannuation Taxation
Telecommunications Tourism Transportation Welfare system Society Crime Demographics Education Ethnic groups Healthcare Homelessness Immigration Languages
Religion Women Culture Architecture Art Arts Australian English Cinema Cuisine Dance Literature Music Public holidays Radio Sport Symbols Television Theatre
Outline Index Book Portal Retrieved from `` https://en.
</item 5>

<item 6>
List of Australian capital cities
Jump up ^ in 1911 , South Australia transferred responsibility for governing the Northern Territory to the Commonwealth Government of Australia . It was granted
self - governance in 1978 , however the Northern Territory is still considered a territory under Commonwealth rule , not a state . References ( edit ) Jump up ^
Norfolk Island - Kingston http://www.pitcairners.org/vt_kingston.html Norfolk Island 's Home on the Web Jump up ^
http://www.abs.gov.au/AUSSTATS/abs@.nsf/Lookup/3218.0Main+Features12016?OpenDocument Jump up ^ http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/3218.
</item 6>

<item 7>
Flinders Street railway station
Collins Lane Public museums ACCA ACMI Chinese Hellenic Ian Potter Immigration Melbourne Observatory NGV Australia NGV International Old Melbourne Gaol Old
Treasury Building RMIT Gallery Institutions Government House Town Hall Parliament House State Library Supreme Court Victoria Barracks Notable structures Arts
Centre Eureka Tower Federation Square Melbourne Star Royal Exhibition Building Shrine of Remembrance St Patrick 's Cathedral St Paul 's Cathedral Sports venues
Docklands ( Etihad ) Stadium Grand Prix Circuit Icehouse Lakeside Stadium MCG Melbourne Park ( Margaret Court Arena - Multi-Purpose Venue ( Hisense Arena ) -
Rod Laver Arena ) Rectangular Stadium ( AAMI Park ) Sports and Aquatic Centre Sports and Entertainment ( Holden ) Centre Royal Park Golf Club State Netball and
Hockey Centre Parks and gardens Albert Park Alexandra Gardens Birrarung Marr Carlton Gardens Fitzroy Gardens Flagstaff Gardens Kings Domain Queen Victoria
Gardens Royal Botanic Gardens Royal Park Treasury Gardens Yarra Park Transport Bolte Bridge Capital City Trail City Circle Tram CityLink City Loop Flinders
Street station Melbourne Central station Southern Cross station Trams West Gate Bridge Yarra River See also : Lanes and arcades of Melbourne , List of museums
in Melbourne , List of theatres in Melbourne , and Parks and gardens of Melbourne Public Transport Victoria
</item 7>

<item 8>
List of Australian capital cities
WESTERN AUSTRALIA NORTHERN TERRITORY SOUTH AUSTRALIA QUEENSLAND NEW SOUTH WALES VICTORIA TASMANIA Perth Adelaide Melbourne Canberra ( ACT ) Sydney Brisbane
Hobart Darwin State and Territory Capitals of Australia Jurisdiction Capital City Population State / Territory Population Date of statehood Capital since Image
Australia and Australian Capital Territory Canberra 435,019 403,468 Must remain a territory as per s.
</item 8>

<item 9>
Culture of Australia
While the major Australian cities enjoyed the boom of the Victorian era , the Australian gold rushes of the mid-19th century brought major construction works
and exuberant Victorian architecture to the major cities , particularly Melbourne , and regional cities such as Ballarat and Bendigo . Other significant
architectural movements in Australian architecture include the Federation style at the turn of the 20th century , and the modern styles of the late 20th century
which also saw many older buildings demolished . The Queenslander is a term which denotes the primarily residential style of warm climate architecture developed
in Queensland and northern parts of New South Wales . Religious architecture is also prominent throughout Australia , with large Anglican and Catholic
cathedrals in every major city and Christian churches in most towns . Notable examples include St Patrick 's Cathedral , Melbourne and St Mary 's Cathedral ,
Sydney . Other houses of worship are also common , reflecting the cultural diversity existing in Australia ; the oldest Islamic structure in the Southern
Hemisphere is the Central Adelaide Mosque ( built in the 1880s ) , and one of the Southern Hemisphere 's largest Buddhist Temples is Wollongong 's Nan Tien
Temple .
</item 9>

<item 10>
List of Australian capital cities
wikipedia.org/w/index.php?title=List_of_Australian_capital_cities&oldid=813441772 '' Categories : Australian capital cities Demographics of Australia Lists of
capitals of country subdivisions Lists of cities in Australia Talk About Wikipedia Languages Bahasa Indonesia ქართული Македонски Bahasa Melayu Norsk Português
Русский தமிழ் اردو Edit links This page was last edited on 3 December 2017 , at 19 : 08 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=932aa3f46c9223d1 ===

=== ITEM id=a810eede58b9efe2 ===
Gold sentences and facts:
Sentence 0: Prior to 1940 the term `` jeep '' had been used as U.S. Army slang for new recruits or vehicles , but the World War II `` jeep '' that went into production in 1941 specifically tied the name to this light military 4x4 , arguably making them the oldest four - wheel drive mass - production vehicles now known as SUVs . The Jeep became the primary light 4 - wheel - drive vehicle of the United States Army and the Allies during World War II , as well as the postwar period . The term became common worldwide in the wake of the war . Doug Stewart noted : `` The spartan , cramped , and unstintingly functional jeep became the ubiquitous World War II four - wheeled personification of Yankee ingenuity and cocky , can - do determination . ''
  Fact 0.0: Before 1940, 'jeep' had been used as U.S. Army slang for new recruits or vehicles
  Fact 0.1: The World War II 'jeep' that went into production in 1941 specifically tied the name to a light military 4x4
  Fact 0.2: The World War II jeep arguably made these vehicles the oldest four-wheel drive mass-production vehicles now known as SUVs
  Fact 0.3: The Jeep became the primary light four-wheel-drive vehicle of the United States Army during World War II
  Fact 0.4: The Jeep became the primary light four-wheel-drive vehicle of the Allies during World War II
  Fact 0.5: The Jeep was the primary light four-wheel-drive vehicle of the United States Army and the Allies in the postwar period
  Fact 0.6: The term 'jeep' became common worldwide in the wake of the war
  Fact 0.7: Doug Stewart noted that the spartan, cramped, and unstintingly functional jeep became the ubiquitous World War II four-wheeled personification of Yankee ingenuity and cocky, can-do determination
Sentence 1: At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are suitable for use on rough terrain .
  Fact 1.0: At least two Jeep models, the CJ-5 and the SJ Wagoneer, enjoyed extraordinary three-decade production runs
  Fact 1.1: The CJ-5 and the SJ Wagoneer each had a production run of a single body generation
  Fact 1.2: Jeeps have inspired a number of other light utility vehicles since the war, such as the Land Rover
  Fact 1.3: Many Jeep variants serving similar military and civilian roles have since been designed in other nations
  Fact 1.4: In lowercase, the term 'jeep' continues to be used as a generic term
  Fact 1.5: The generic term 'jeep' refers to vehicles inspired by the Jeep that are suitable for use on rough terrain
Sentence 2: Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep 's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in 2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone , over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth between $22 and $33.5 billion -- slightly more than all of FCA ( US ) .
  Fact 2.0: Jeep's current product range consists solely of sport utility vehicles and off-road vehicles
  Fact 2.1: Jeep's product range has also included pickup trucks in the past
  Fact 2.2: Some of Jeep's vehicles, such as the Grand Cherokee and the Wagoneer, reach into the luxury segment
  Fact 2.3: Jeep sold 1.4 million SUVs globally in 2016
  Fact 2.4: Jeep sold 500,000 SUVs globally in 2008
  Fact 2.5: Two-thirds of the 1.4 million SUVs Jeep sold globally in 2016 were sold in North America
  Fact 2.6: Jeep was Fiat-Chrysler's best selling brand in the U.S. during the first half of 2017
  Fact 2.7: In the U.S. alone, over 2400 dealerships hold franchise rights to sell Jeep-branded vehicles
  Fact 2.8: If Jeep were spun off into a separate company, it is estimated to be worth between $22 and $33.5 billion
  Fact 2.9: That estimated value is slightly more than all of FCA (US)

Retrieved items:
<item 1>
Jeep
Prior to 1940 the term `` jeep '' had been used as U.S. Army slang for new recruits or vehicles , but the World War II `` jeep '' that went into production in
1941 specifically tied the name to this light military 4x4 , arguably making them the oldest four - wheel drive mass - production vehicles now known as SUVs .
The Jeep became the primary light 4 - wheel - drive vehicle of the United States Army and the Allies during World War II , as well as the postwar period . The
term became common worldwide in the wake of the war . Doug Stewart noted : `` The spartan , cramped , and unstintingly functional jeep became the ubiquitous
World War II four - wheeled personification of Yankee ingenuity and cocky , can - do determination . '' The Jeep marque has been headquartered in Toledo , Ohio
, ever since Willys - Overland launched production of the first CJ or Civilian Jeep branded models there in 1945 . Its replacement , the conceptually consistent
Jeep Wrangler series , remains in production since 1986 .
</item 1>

<item 2>
Jeep
Off - road abilities Jeep Wrangler off - roading . Jeep advertising has always emphasized the brand 's vehicles ' off - road capabilities . Today , the Wrangler
is one of the few remaining four - wheel - drive vehicles with solid front and rear axles . These axles are known for their durability , strength , and
articulation . New Wranglers come with a Dana 44 rear differential and a Dana 30 front differential . The upgraded Rubicon model of the JK Wrangler is equipped
with electronically activated locking differentials , Dana 44 axles front and rear with 4.10 gears , a 4 : 1 transfer case , electronic sway bar disconnect and
heavy duty suspension . Another benefit of solid axle vehicles is they tend to be easier and cheaper to `` lift '' with aftermarket suspension systems . This
increases the distance between the axle and chassis of the vehicle . By increasing this distance , larger tires can be installed , which will increase the
ground clearance , allowing it to traverse even larger and more difficult obstacles . In addition to higher ground clearance , many owners aim to increase
suspension articulation or `` flex '' to give their Jeeps greatly improved off - road capabilities .
</item 2>

<item 3>
Jeep
2 - door version ; produced by AEV with the Jeep logo . 2007 -- 2018 Wrangler JK 2007 -- 2009 JK Rubicon , Sahara , X 2010 JK Rubicon , Sahara , Mountain ,
Islander , and Sport Editions 2011 70th Anniversary , Call of Duty : Black Ops , Mojave , and Oscar Mike Military ( 200 made ) Editions 2011 JK - 8 Independence
-- a MOPAR JK pick - up conversion kit , mirroring the 1980s CJ - 8 Scrambler pick - up 2013 Rubicon 10th Anniversary Edition 2013 - 2017 Brute Double Cab :
Pickup truck , 4 - door version , produced by American Expedition Vehicles 2014 Willys Wheeler Edition 2017 -- Jeep Wrangler JL Grand Cherokee Main article :
Jeep Grand Cherokee 1st generation Grand Cherokee ZJ . 2008 - 2010 WK Grand Cherokee .
</item 3>

<item 4>
Jeep
Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep
's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in
2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone ,
over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth
between $22 and $33.5 billion -- slightly more than all of FCA ( US ) . Prior to 1940 the term `` jeep '' had been used as U.S.
</item 4>

<item 5>
Jeep
Segar , as early as mid-March 1936 . Eugene the Jeep was Popeye 's `` jungle pet '' and was `` small , able to move between dimensions and could solve seemingly
impossible problems . '' The word `` jeep '' however , was used as early as World War I , as US Army slang for new uninitiated recruits , or by mechanics to
refer to new unproven vehicles . In 1937 , tractors which were supplied by Minneapolis Moline to the US Army were called jeeps . A precursor of the Boeing B -
17 Flying Fortress was also referred to as the jeep . Words of the Fighting Forces by Clinton A. Sanders , a dictionary of military slang , published in 1942 ,
in the library at The Pentagon gives this definition : Jeep : A four - wheel drive vehicle of one - half - to one - and - one - half - ton capacity for
reconnaissance or other army duty . A term applied to the bantam - cars , and occasionally to other motor vehicles ( U.S.A. ) in the Air Corps , the Link
Trainer ; in the armored forces , the 1⁄2 - ton command vehicle .
</item 5>

<item 6>
Jeep
Jeep - wikipedia Jeep Jump to : navigation , search This article is about the U.S. motor manufacturing company . For the original World War II Jeep , see Willys
MB . For other uses , see Jeep ( disambiguation ) . Jeep Type Division Industry Automotive Founded 1943 trademark application 1945 first civilian Jeep launched
Headquarters Toledo , Ohio , U.S. Area served Worldwide Key people Michael Manley ( CEO of Jeep division ) Sergio Marchionne ( CEO of FCA US LLC ) Products
Sport utility vehicles Luxury vehicles Owner Fiat Chrysler Automobiles Parent Willys - Overland ( 1943 - 1953 ) Kaiser - Jeep ( 1953 - 1970 ) American Motors
Corporation ( 1970 - 1986 ) Chrysler ( 1987 - present ) Website jeep.com Jeep is a brand of American automobiles that is a division of FCA US LLC ( formerly
Chrysler Group , LLC ) , a wholly owned subsidiary of Fiat Chrysler Automobiles . Jeep has been a part of Chrysler since 1987 , when Chrysler acquired the Jeep
brand , along with remaining assets , from its previous owner : American Motors Corporation ( AMC ) .
</item 6>

<item 7>
Jeep
Also referred to as `` any small plane , helicopter , or gadget . '' This definition is supported by the use of the term `` jeep carrier '' to refer to the Navy
's small escort carriers . Early in 1941 , Willys - Overland demonstrated the vehicle 's off - road capability by having it drive up the steps of the United
States Capitol , driven by Willys test driver Irving `` Red '' Hausmann , who had recently heard soldiers at Fort Holabird calling it a `` jeep . '' When asked
by syndicated columnist Katharine Hillyer for the Washington Daily News ( or by a bystander , according to another account ) what it was called , Hausmann
answered , `` It 's a jeep . '' Katharine Hillyer 's article was published nationally on February 19 , 1941 , and included a picture of the vehicle with the
caption : LAWMAKERS TAKE A RIDE - With Senator Meade , of New York , at the wheel , and Representative Thomas , of New Jersey , sitting beside him , one of the
Army 's new scout cars , known as `` jeeps '' or `` quads '' , climbs up the Capitol steps in a demonstration yesterday .
</item 7>

<item 8>
Jeep
Please consider summarizing the material while citing sources as needed . ( December 2017 ) Main article : List of Jeep vehicles Jeep CJ Main article : Jeep CJ
( Civilian Jeep ) 1982 Jeep Scrambler The CJ ( for `` Civilian Jeep '' ) series were literally the first `` Jeep '' branded vehicles produced for the civilian
public , beginning in 1945 with the CJ - 2A , followed by the CJ - 3A in 1949 and the CJ - 3B in 1953 . These early Jeeps are commonly referred to as ``
flatfenders '' because their front fenders were flat and rectangular across , like on their military forebears , the Willys MB and identical Ford GPW models .
The CJ - 4 exists only as a 1951 prototype , and is the `` missing link '' between the flat - fendered CJ - 2A and CJ - 3B and the round - fendered CJ - 5 first
introduced in 1955 .
</item 8>

<item 9>
Jeep
5 Main articles : Jeep Cherokee ( XJ ) and Jeep Comanche 1984 -- 2001 XJ Cherokee 1984 -- 1990 XJ Wagoneer 1984 -- 1985 Broughwood 1984 -- 1990 Limited 1986 --
1992 Jeep Comanche ( MJ ) Jeep Wrangler 1989 Jeep Wrangler YJ Islander 1997 Jeep Wrangler TJ 2016 Jeep Wrangler JK Unlimited ( MIAS ' 16 ) Jeep Wrangler Main
article : Jeep Wrangler 1987 -- 1995 Jeep Wrangler YJ 1991 -- 1993 Renegade 1988 -- 1995 Wrangler Long -- Venezuela 1995 Wrangler Rio Grande 1997 -- 2006
Wrangler TJ 2002 TJ Se , X , Sport , Sahara models 2003 TJ Rubicon , Rubicon Tomb Raider Edition , Sahara , Sport , X , Se models , Freedom Edition 2004 -- 2006
TJ Long Wheel Base ( LJ ) Unlimited ( 15 '' Longer than a standard TJ ) Rubicon , Sport , X , Se models 2004 -- 2005 Willys Edition ( 2004 -- 1997 made ,
</item 9>

<item 10>
Jeep
With its solid axles and open top , the Wrangler has been called the Jeep model that is as central to the brand 's identity as the rear - engined 911 is to
Porsche . At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps
have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles
have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are
suitable for use on rough terrain . Contents ( hide ) 1 World War II Jeeps 1.1 Development -- 1 . Bantam Reconnaissance Car 1.2 Development -- 2 . Enter Willys
and Ford 1.3 Full production -- Willys MB and Ford GPW 2 Post-war military Jeeps 2.1 CJ - V35 / U 2.2 M715 3 Jeep etymology 4 Brand , trademarks and image 4.1
Off - road abilities 5 Company history and ownership 5.
</item 10>
=== END ITEM id=a810eede58b9efe2 ===

=== ITEM id=5ca335208e389dd5 ===
Gold sentences and facts:
Sentence 0: The Quebec Winter Carnival or the Carnaval de Québec is a pre-Lenten festival held in Quebec City . After being held intermittently since 1894 , the Carnaval de Québec has been celebrated annually since 1955 . That year Bonhomme Carnaval , the mascot of the festival , made his first appearance . Up to one million people attended the Carnaval de Québec in 2006 making it the largest winter festival in the world .
  Fact 0.0: The Quebec Winter Carnival (Carnaval de Québec) is a pre-Lenten festival.
  Fact 0.1: The Carnaval de Québec is held in Quebec City.
  Fact 0.2: The Carnaval de Québec was held intermittently after 1894.
  Fact 0.3: The Carnaval de Québec has been celebrated annually since 1955.
  Fact 0.4: In 1955, Bonhomme Carnaval, the mascot of the festival, made his first appearance.
  Fact 0.5: Up to one million people attended the Carnaval de Québec in 2006.
  Fact 0.6: In 2006, the Carnaval de Québec was the largest winter festival in the world.

Retrieved items:
<item 1>
God Save the Queen
Ottawa : Queen 's Printer for Canada . p. 2 . ISBN 978 - 1 - 100 - 12739 - 2 . Retrieved 16 January 2011 . Jump up ^ Alberta Police and Peace Officers '
Memorial Day 2009 Order of Service , Queen 's Printer for Alberta , 27 September 2009 access - date = requires url = ( help ) Jump up ^ Department of Canadian
Heritage . `` Ceremonial and Canadian Symbols Promotion > The National Flag of Canada '' . Queen 's Printer for Canada . Retrieved 5 July 2010 . Jump up ^ Royal
Canadian Legion Dominion Command ( 4 November 2009 ) . `` National Remembrance Day Ceremony '' . Royal Canadian Legion . Archived from the original on 12 March
2010 . Retrieved 5 July 2010 . Jump up ^ Department of Veterans Affairs . `` Canada Remembers > Partnering Opportunities > Community Engagement Partnership Fund
> Nova Scotia > Community Engagement Partnership Fund : Nova Scotia '' . Queen 's Printer for Canada . Retrieved 5 July 2010 .
</item 1>

<item 2>
Quebec Winter Carnival
An Ice canoe race on the St. Lawrence river . The Québec International Pee - Wee Hockey Tournament ( not officially part of the carnival program since 1977 ) .
The Snowboard World Cup in Quebec City ( not part of the carnival official program ) . Also not part of the official program but worth a visit or a stay , the
Quebec City Ice Hotel is open every year from early January to late March with its bar , nightclub , exhibition galleries and ceremonial chapel . Gallery ( edit
) Ice sculpture ( 2005 ) Family venue ( 2006 ) Ice Palace ( 2009 ) Parade in Montreal ( 2011 ) Canoe race ( 2011 ) Ice Palace ( 2011 ) Toboggan at Terrasse
Dufferin References ( edit ) Jump up ^ Daniel Shafto ( 1 January 2009 ) . Carnival . Infobase Publishing . p. 80 . ISBN 978 - 1 - 4381 - 2660 - 9 . Jump up ^
Hilary Davidson ; Paul Karr ; Herbert Bailey Livesey ; Bill McRae ; Donald Olson ( 14 August 2006 ) .
</item 2>

<item 3>
Quebec Winter Carnival
Frommer 's Canada : With the best hiking & outdoor adventures . John Wiley & Sons . p. 300 . ISBN 978 - 0 - 470 - 04457 - 5 . Jump up ^
https://www.theglobeandmail.com/life/travel/activities-and-interests/the-snowman-behind-the-legend-get-to-know-quebecs-beloved-bonhomme/article37716316/ Quebec
Winter Carnaval virtual tour in panoramic photography Tornoipee-wee.qc.ca Fissnowboard.com Icehotel-canada.com Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Quebec_Winter_Carnival&oldid=822352977 '' Categories : Tourist attractions in Quebec City Festivals in Quebec City
Winter festivals in Canada Mardi Gras 1894 establishments in Quebec Hidden categories : Articles needing additional references from January 2018 All articles
needing additional references Quebec articles missing geocoordinate data All articles needing coordinates Talk Contents About Wikipedia Deutsch Español Français
한국어 日本 語 Português Svenska Edit links This page was last edited on 25 January 2018 , at 21 : 54 .
</item 3>

<item 4>
Quebec Winter Carnival
Free outdoor public banquets ( brunch , breakfast , etc . ) . . The Canadian , Québécois , International and Student artist snow sculpture contests on the
Plains of Abraham , the main setting of the carnival . The Plains are a public city park and stay open for leisure activities , including snowshoeing and
cross-country skiing trails , during carnival time . Part of the Plains around the Citadel is transformed into an outdoor winter amusement park with various
family - themed activities , including the display of the three main snow - sculpting contests ( Canada 's provinces , Quebec 's regions , International ) and
the traditional bikini snow bath event ( bain de neige ) Outdoor dance parties are held at the Ice Palaces . Kiosks and other outlets in the city sell the
Bonhomme effigy tag that grants admission into most of the events , although some are free outside the main site . Most commercial main streets are decorated
and some bars and restaurants set up a winter patio in front of their establishments . Bonhomme -- short for bonhomme de neige ( `` snowman '' ) is the official
ambassador of the festivities , the castle lord of the Ice Palace .
</item 4>

<item 5>
Quebec Winter Carnival
Quebec winter carnival - wikipedia Quebec winter carnival Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2018 ) ( Learn how and when to
remove this template message ) Quebec Winter Carnival Carnaval de Québec Bonhomme Carnaval in 2011 Genre Festival Begins End of January Ends Mid February
Frequency Annually Location ( s ) Quebec City Country Canada Website Carnaval de Québec The Quebec Winter Carnival or the Carnaval de Québec is a pre-Lenten
festival held in Quebec City . After being held intermittently since 1894 , the Carnaval de Québec has been celebrated annually since 1955 . That year Bonhomme
Carnaval , the mascot of the festival , made his first appearance . Up to one million people attended the Carnaval de Québec in 2006 making it the largest
winter festival in the world .
</item 5>

<item 6>
Quebec Winter Carnival
Contents ( hide ) 1 Activities and attractions 2 Feasts and restaurants 3 Races and tournaments 4 Gallery 5 References Activities and attractions ( edit ) The
most famous attractions of this winter festival are the night - time and daytime parades led by mascot Bonhomme Carnaval . The parades wind through the upper
city , decorated for the occasion with lights and ice sculptures . Numerous public and private parties , shows and balls are held across the city , some of them
outside in the bitter cold , testimony to the Québécois ' fabled joie de vivre . Other major events include : A masquerade ball with up to 400 participants at
the grand ballroom of the Château Frontenac . The opening and closing ceremonies taking place at the Ice Palace before thousands of participants , Bonhomme and
the mayor of Quebec . Outdoor sport events ( snowboarding , ice canoe , snowshoes , hockey , dog - sledding , etc. , some of them part of World Championship
tournaments ) inside and outside the city . Free outdoor public banquets ( brunch , breakfast , etc . ) . .
</item 6>

<item 7>
God Save the Queen
Queen 's Printer for Canada . Retrieved 5 July 2010 . Jump up ^ Remembrance Day ( PDF ) , Vancouver Board of Parks and Recreation , 11 November 2009 , archived
from the original ( PDF ) on 7 July 2011 , retrieved 5 July 2010 Jump up ^ `` Honours and Salutes '' . Queen 's Printer for Canada . Retrieved 4 March 2017 . ^
Jump up to : Bélanger , Claude . `` The Quebec History Encyclopedia '' . In Marianopolis College . National Anthem of Canada . Montreal : Marianopolis College .
Retrieved 5 July 2010 . ^ Jump up to : Department of Canadian Heritage . `` National Anthem : O Canada > Parliamentary Action '' . Queen 's Printer for Canada .
Retrieved 29 June 2010 . ^ Jump up to : Department of National Defence ( 1 April 1999 ) , The Honours , Flags and Heritage Structure of the Canadian Forces (
PDF ) , Ottawa : Queen 's Printer for Canada , p.
</item 7>

<item 8>
Quebec Winter Carnival
Bonhomme is described as a seven - foot - tall , four - hundred pound snowman sporting a red cap , black buttons and a ceinture fléchée . It is traditional to
drink Caribou , a hot alcoholic beverage , to keep warm . The public auction is a fundraising event in aid of the carnival . This auction features many goods
and services donated for silent auction and live auction . Feasts and restaurants ( edit ) The Business Leaders ' Luncheon , organized by the Québec City
Chamber of Commerce . The Calgary Stampede Flap Jack Breakfast . In 2008 , carnival - goers were served by Prime Minister Stephen Harper . The ' Restaurant
Partners ' Campaign is a 179 - day promotion during which Québec City restaurants offer customers a special menu for a fixed price throughout the carnival (
including appetizer , soup , or salad , a main course , and a dessert ) . Races and tournaments ( edit ) A sleigh race in which drivers and their horses take
part in a single - and double - harness race on the Plains of Abraham . An Ice canoe race on the St. Lawrence river .
</item 8>

<item 9>
Quebec Winter Carnival
About Wikipedia
</item 9>

<item 10>
Graph query row:
bonhomme: Bonhomme Carnaval
carnival: Quebec Winter Carnival
relationship: RELATED_TO
role: mascot
relationship_detail: None
relation: None
context: None
fact: None
result: None
outcome: None
</item 10>
=== END ITEM id=5ca335208e389dd5 ===

=== ITEM id=4034ffec587872dc ===
Gold sentences and facts:
Sentence 0: It was confirmed on 25 January 2018 , that Dancing on Ice had been recommissioned for an eleventh series to air in 2019 .
  Fact 0.0: On 25 January 2018, it was confirmed that Dancing on Ice had been recommissioned.
  Fact 0.1: Dancing on Ice was recommissioned for an eleventh series.
  Fact 0.2: The eleventh series is to air in 2019.

Retrieved items:
<item 1>
Dancing on Ice
The show consists of celebrities and their professional partners figure skate in front of a panel of judges . The series , broadcast on ITV , started on 14
January 2006 and ended on 9 March 2014 after the show 's contract was not renewed by ITV . On 4 September 2017 , it was announced that a revived series would
air on ITV from 7 January 2018 with Schofield and Willoughby returning as hosts alongside a new backstage host , Jordan Banjo . It was confirmed on 25 January
2018 , that Dancing on Ice had been recommissioned for an eleventh series to air in 2019 . Contents ( hide ) 1 Background and history 1.1 Cancellation and
revival 2 Format 3 Series overview 4 Presenters , judges and coaches 5 Professionals 6 Required elements 7 Main series results 7.1 Series 1 ( 2006 ) 7.2 Series
2 ( 2007 ) 7.3 Series 3 ( 2008 ) 7.4 Series 4 ( 2009 ) 7.5 Series 5 ( 2010 ) 7.
</item 1>

<item 2>
Dancing on Ice
The presenters , judges and participating celebrities are yet to be confirmed by ITV . Series averages All information in this table comes from BARB . including
figures from ITV + 1 and ITV HD channels Series Day # Ep . Premiere Final Aired Average viewers ( in millions ) Date Viewers ( in millions ) Date Viewers ( in
millions ) Saturdays 16 14 January 2006 9.83 4 March 2006 11.68 2006 9.12 18 20 January 2007 9.08 17 March 2007 9.14 2007 8.00 Sundays 20 13 January 2008 9.34
16 March 2008 12.08 2008 8.93 22 11 January 2009 8.83 22 March 2009 11.31 2009 8.86 5 23 10 January 2010 9.64 28 March 2010 9.42 8.28 6 23 9 January 2011 10.23
27 March 2011 9.35 2011 8.
</item 2>

<item 3>
Dancing on Ice
the show was renamed following the failure of ITV 's celebrity oriented 2005 summer schedule . Dancing on Ice is frequently compared to the BBC 's Strictly Come
Dancing . In 2004 , the BBC aired a Strictly special entitled Strictly Ice Dancing at Christmas , which was won by England goalkeeper David Seaman ( who was a
contestant in series 1 of Dancing on Ice ) . ITV 's show was given a January premiere amidst network doubts about its viability but became a surprise hit in
Britain , where it became the third highest rated television show of 2006 . It attained an impressive 13 million viewers for the final in March . Britain 's
best - known ice - skating duo and former Olympic champions Jayne Torvill and Christopher Dean help to train the aspiring dancers , and also appear throughout
the show with comments and advice . Head Coach Karen Barber also trains the skaters for the live show . From the beginning , Torvill and Dean opened every
episode with a performance , with the exception of the second , third and fourth shows of the fourth series , when Torvill performed alone due to Dean 's
recovery from a shoulder operation .
</item 3>

<item 4>
Dancing on Ice
Radio Times . 25 January 2018 . Retrieved 27 January 2018 . Jump up ^ `` Dancing on Ice series 3 '' . Digital Spy. 21 May 2007 . Retrieved 21 May 2006 . Jump up
^ `` About the show '' . itv.com. 5 December 2007 . Retrieved 30 December 2007 . Jump up ^ `` Dancing on Ice -- TV3 '' . Tv3.ie . Archived from the original on
15 March 2011 . Retrieved 27 March 2011 . Jump up ^ `` Dancing on Ice dates confirmed News Phillip Schofield '' . Officialphillipschofield.com . Retrieved 24
December 2012 . Jump up ^ `` ' Dancing on Ice ' lineup revealed : Heidi Range , Jennifer Ellison , more -- Dancing on Ice News -- Reality TV '' . Digital Spy. 3
January 2012 . Retrieved 24 December 2012 . Jump up ^ `` Chesney 's ice exit '' . itv.com. 4 January 2012 .
</item 4>

<item 5>
Dancing on Ice
23 27 March 2011 9.35 2011 8.15 7 23 8 January 2012 8.42 25 March 2012 7.01 2012 6.63 8 19 6 January 2013 7.49 10 March 2013 7.36 2013 6.32 9 19 5 January 2014
6.76 9 March 2014 5.98 2014 5.04 10 10 7 January 2018 8.56 11 March 2018 2018 Awards Year Group Award Result Reference ( s ) 2006 National Television Awards
Most Popular Entertainment Programme Nominated 2007 National Television Awards Most Popular Talent Show Nominated 2008 National Television Awards Most Popular
Talent Show Nominated TV Quick Awards Best Talent Show Nominated 2009 National Television Awards Most Popular Talent Show Nominated TV Quick Awards Best Talent
Show Nominated RTS Craft & Design Award Best Costume Design Won RTS Craft & Design Award Best Multicamera Work Nominated RTS Craft & Design Award Best Costume
Design Nominated 2011 National Television Awards Most Popular Talent Show Nominated RTS
</item 5>

<item 6>
Dancing on Ice
of series 10 No. of episodes 138 Production Executive producer ( s ) Jane Beacon Katie Rawcliffe Location ( s ) Elstree Studios ( 2006 -- 10 , 2012 -- 14 )
Shepperton Studios ( 2011 ) RAF Bovingdon ( 2018 -- ) Running time 60 -- 120 minutes Production company ( s ) ITV Studios Release Original network ITV Picture
format 16 : 9 1080i ( HDTV ) 16 : 9 576i ( SDTV ) Original release Original Series : 14 January 2006 ( 2006 - 01 - 14 ) -- 9 March 2014 ( 2014 - 03 - 09 )
Revived Series : 7 January 2018 ( 2018 - 01 - 07 ) -- present External links Website Dancing on Ice is a British television show presented by Phillip Schofield
alongside Holly Willoughby from 2006 to 2011 , and Christine Bleakley from 2012 to 2014 . The show consists of celebrities and their professional partners
figure skate in front of a panel of judges .
</item 6>

<item 7>
Dancing on Ice
Later that month Gardiner confirmed that he would return to the judging panel . Ashley 's brother Jordan Banjo acts as the show 's backstage digital host . ITV
Racing 's betting presenter Matt Chapman was announced as the new commentator on 16 December 2017 . However , following the first live show on 7 January ,
Chapman claimed that he had `` quit '' the show but it was later confirmed to be a decision by the ITV producers , responding to pressure from the TV audience .
Sport commentator Sam Matterface was later confirmed as Chapman 's replacement . It was announced on 25 January 2018 that Dancing on Ice would return for an
eleventh series in 2019 . The presenters , judges and participating celebrities are yet to be confirmed by ITV . Format Each week the celebrities and their
partners perform a live ice dance routine . The four / five ( as of Series 8 ) judges ( commonly known as the Ice Panel ) judge each performance and give a mark
between 0.0 and 10.0 ( 0.0 to 6.0 between series 1 and 5 ) , depending on the performance .
</item 7>

<item 8>
Dancing on Ice
5 , 8.5 , 9.0 = 26.0 Eliminated Tessa Sanderson CBE Javelin Gold ( LA 1984 ) Yannick Bonheur 7.5 , 7.5 , 8.0 = 23.0 Eliminated The judges scores were added to
the studio audiences votes to decide the winner . No public vote took place , as the programme was recorded earlier in the year . Dancing on Ice Friday ( 2010 )
A brand new spin - off show for the 2010 series of Dancing on Ice , appropriately called Dancing on Ice Friday , gave viewers the insight to the training of the
celebrities over the last week . It was presented by television presenter Ben Shephard and former contestant and Loose Women star Coleen Nolan . The show was
broadcast from 8 pm to 8.30 pm on Friday evenings on ITV throughout the duration of the main shows season . STV who broadcast the main show did not broadcast
this on the Friday evening but after repeating the previous week 's main show on the following Saturday afternoon . Due to poor ratings , Dancing on Ice Friday
was axed prior to the 2011 series .
</item 8>

<item 9>
Dancing on Ice
Dancing on Ice - wikipedia Dancing on Ice Jump to : navigation , search This article is about the British series . For elsewhere , see Dancing on Ice around the
world . For the most recent series , see Dancing on Ice ( series 10 ) . Dancing on Ice Title -- card used for the 2018 revival Created by ITV Studios Creative
director ( s ) Christopher Dean ( 2006 -- 2014 ) Jayne Torvill ( 2006 -- 2014 ) Presented by Phillip Schofield Holly Willoughby Christine Bleakley Jordan Banjo
( backstage ) Judges Robin Cousins Karen Barber Jason Gardiner Karen Kresge Nicky Slater Natalia Bestemianova Ruthie Henshall Emma Bunton Louie Spence Katarina
Witt Ashley Roberts Christopher Dean Jayne Torvill Ashley Banjo Voices of Tony Gubba ( 2006 -- 13 ) Simon Reed ( 2013 -- 14 ) Matt Chapman ( 2018 ) Sam
Matterface ( 2018 -- ) Theme music composer Paul Farrer Country of origin United Kingdom Original language ( s ) English No. of series 10 No.
</item 9>

<item 10>
Dancing on Ice
As part of ITV 's new revamped schedule at the start of 2008 , from the third series , the show aired on a Sunday night and featured updated music and new
titles as well as redesigned graphics . The series 3 finale was a massive draw , pulling an average 11.7 million viewers ( up from 9.6 million the previous year
) peaking at 12.6 million viewers over the two - hour slot , up over a million from series 2 . The seventh series began on 8 January 2012 , with Torvill and
Dean as coaches for which they will be paid £ 250,000 each series . Christine Bleakley was announced as the new co-presenter of the show replacing Holly
Willoughby , whilst Katarina Witt and Louie Spence replaced Emma Bunton and Jason Gardiner on the judging panel . On 20 November 2012 , it was announced that
Gardiner was to return to the panel for the 2013 series , replacing Spence .
</item 10>
=== END ITEM id=4034ffec587872dc ===
